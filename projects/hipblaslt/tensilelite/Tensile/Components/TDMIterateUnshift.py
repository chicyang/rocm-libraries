# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Undo the TDM iterate-mode edge shift in the accumulators.

The boundary component's rows land in LDS `delta` too low, so the MI wave that
covers those rows holds results for `global_row = R - delta`. Moving the values
down by `delta` restores the ordinary `coord0`/`coord1` meaning, which leaves
every store address stream -- D, C, E, bias, scaleAlphaVec, scaleA/B, gate --
untouched.

Both source and destination of every move are at or above the shifted region's
start, so the move never reaches below it. The top `delta` destinations receive
values from outside the block; their coordinate is exactly the free size, so the
existing edge mask discards them.

`delta` is a runtime value and assembly has no dynamic register index, so it is
decomposed into powers of two and one straight-line pass is emitted per bit.
That keeps both code size and run time logarithmic in `tile_dim1`, which can be
32 in other configurations.
"""

from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister, DSModifiers, sgpr, vgpr
from rocisa.instruction import (
    DSBPermuteB32,
    SAndB32,
    SCBranchSCC0,
    SCmpEQU32,
    SCSelectB32,
    SLShiftRightB32,
    SMulI32,
    SSubU32,
    SWaitCnt,
    VAndB32,
    VReadfirstlaneB32,
)

from rocisa.functions import vectorStaticDivide, vectorStaticMultiply, vectorStaticRemainder

from ..Component import TDMIterateUnshift as TDMIterateUnshiftBase
from ..KernelWriterModules import accToArchMapper
from ..SolutionStructs import tdm_iterate_edge


class TDMIterateUnshiftMFMA(TDMIterateUnshiftBase):
    """Accumulator un-shift for the TDM iterate-mode free-dimension edge shift."""

    kernel = {"EnableMatrixInstruction": True}

    def __call__(self, writer, kernel, tP) -> Module:
        tc = tP["tensorChar"]
        ti = tP["idx"]
        module = Module("TDMIterateUnshift%s" % tc)
        if not kernel.get("_TDMIterEdgeShift%s" % tc, False):
            return module

        g = tdm_iterate_edge.geometry(kernel, tc)
        lay = tdm_iterate_edge.coalLayout(kernel, tP["isA"])
        steps = tdm_iterate_edge.shift_steps(g["tileDim1"])

        _, arch2acc = accToArchMapper(kernel)

        size = "SizeI" if tP["isA"] else "SizeJ"
        wg = "WorkGroup0" if tP["isA"] else "WorkGroup1"
        mt = kernel["MacroTile%u" % ti]

        deltaSgpr = writer.sgprPool.checkOut(1, tag="unshiftDelta%s" % tc, preventOverflow=False)
        ownerSgpr = writer.sgprPool.checkOut(1, tag="unshiftOwner%s" % tc, preventOverflow=False)
        rowsSgpr = writer.sgprPool.checkOut(1, tag="unshiftRows%s" % tc, preventOverflow=False)

        module.addComment1("TDM iterate edge: un-shift %s accumulators" % tc)
        module.add(SMulI32(sgpr(rowsSgpr), mt, sgpr(wg), "wg * MT(%u)" % mt))
        module.add(SSubU32(dst=sgpr(rowsSgpr), src0=sgpr(size), src1=sgpr(rowsSgpr),
                           comment="rows = Size - wg*MT"))
        writer.tdmIterEdgeDelta(module, kernel, tc, ti, rowsSgpr, deltaSgpr)
        writer.tdmIterEdgeOwner(module, kernel, tc, ti, rowsSgpr, ownerSgpr)

        # `cOwn` names one MI block along the coalesced dimension. A thread holds
        # `miOuterTTCoal` register runs, and consecutive runs of one wave are
        # `miWaveGroupCoal` blocks apart, so the block splits as
        #   tt = cOwn // miWaveGroupCoal   -- which register run to move
        #   waveG0 = cOwn % miWaveGroupCoal -- which wave does the moving
        # and `tt * miWaveGroupCoal + waveG0` reproduces `cOwn`. The oracle
        # guarantees miWaveGroupCoal is a power of two whenever the split is
        # needed, so a shift and a mask suffice.
        miOuterTTCoal = lay["miOuterTTCoal"]
        ttSgpr = None
        if miOuterTTCoal > 1:
            ttSgpr = writer.sgprPool.checkOut(1, tag="unshiftTT%s" % tc, preventOverflow=False)

        # This wave's block index along the coalesced dimension. Waves that do
        # not own the boundary component get delta = 0 and skip every pass, so
        # no exec-mask manipulation is needed.
        miWaveGroupCoal = lay["miWaveGroupCoal"]
        miWGIdStride = (
            kernel["WavefrontSize"]
            if tP["isA"]
            else kernel["WavefrontSize"] * kernel["MIWaveGroup"][0]
        )
        blkVgpr = writer.vgprPool.checkOut(1, "unshiftBlk")
        tmpVgpr = writer.vgprPool.checkOutAligned(2, 2, "unshiftTmp")
        tmpVgprRes = ContinuousRegister(tmpVgpr, 2)
        dummy = writer.vgprPool.checkOut(1, "unshiftDummy")
        module.add(vectorStaticDivide(blkVgpr, "Serial", miWGIdStride, tmpVgprRes))
        with writer.allocTmpSgpr(writer.states.laneSGPRCount, tag="unshiftBlk") as blkSgpr:
            module.add(
                vectorStaticRemainder(dummy, blkVgpr, blkVgpr, miWaveGroupCoal, tmpVgprRes, blkSgpr)
            )
            module.add(VReadfirstlaneB32(dst=sgpr(rowsSgpr), src=vgpr(blkVgpr),
                                         comment="coal block index of this wave"))
        if miOuterTTCoal > 1:
            module.add(SLShiftRightB32(
                dst=sgpr(ttSgpr), shiftHex=(miWaveGroupCoal - 1).bit_length(),
                src=sgpr(ownerSgpr),
                comment="tt = cOwn / miWaveGroupCoal(%u)" % miWaveGroupCoal))
            module.add(SAndB32(sgpr(ownerSgpr), sgpr(ownerSgpr), miWaveGroupCoal - 1,
                               "waveG0 = cOwn %% miWaveGroupCoal(%u)" % miWaveGroupCoal))
        module.add(SCmpEQU32(src0=sgpr(rowsSgpr), src1=sgpr(ownerSgpr),
                             comment="does this wave own the boundary component?"))
        module.add(SCSelectB32(sgpr(deltaSgpr), sgpr(deltaSgpr), 0,
                               "non-owning waves shift by 0"))

        # ds_bpermute address: this lane's own index, scaled to bytes. The
        # DSModifiers offset then selects the neighbour along the coal dim.
        permVgpr = writer.vgprPool.checkOut(1, "unshiftPerm")
        module.add(VAndB32(dst=vgpr(permVgpr), src0=kernel["WavefrontSize"] - 1,
                           src1=vgpr("Serial"), comment="lane index"))
        with writer.allocTmpSgpr(1, tag="unshiftPermMul") as permTmp:
            module.add(vectorStaticMultiply(vgpr(permVgpr), vgpr(permVgpr), writer.states.bpr,
                                            permTmp, comment="lane index -> byte address"))

        # Register indices are compile-time and `tt` is not, so one block of
        # passes is emitted per run and the runtime `tt` selects between them.
        # Exactly one block runs; a non-owning wave has delta = 0 and skips
        # every pass inside whichever block it enters.
        for tt in range(miOuterTTCoal):
            ttSkip = None
            if miOuterTTCoal > 1:
                ttSkip = Label(
                    writer.labels.getNameInc("TDMIterUnshift%s_tt%u" % (tc, tt)), "")
                module.add(SCmpEQU32(src0=sgpr(ttSgpr), src1=tt,
                                     comment="is the boundary component in register run %u?" % tt))
                module.add(SCBranchSCC0(labelName=ttSkip.getLabelName(),
                                        comment="skip register run %u" % tt))
            for s in steps:
                skip = Label(writer.labels.getNameInc("TDMIterUnshift%s_skip%u" % (tc, s)), "")
                module.add(SAndB32(sgpr(rowsSgpr), sgpr(deltaSgpr), s,
                                   "delta & %u ?" % s))
                module.add(SCBranchSCC0(labelName=skip.getLabelName(),
                                        comment="skip the shift-by-%u pass" % s))
                module.add(self._shiftBy(writer, kernel, lay, arch2acc, s, permVgpr, tt))
                module.add(skip)
            if ttSkip is not None:
                module.add(ttSkip)

        if ttSgpr is not None:
            writer.sgprPool.checkIn(ttSgpr)
        writer.vgprPool.checkIn(permVgpr)
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(blkVgpr)
        writer.sgprPool.checkIn(rowsSgpr)
        writer.sgprPool.checkIn(ownerSgpr)
        writer.sgprPool.checkIn(deltaSgpr)
        return module

    def _shiftBy(self, writer, kernel, lay, arch2acc, s: int, permVgpr: int,
                 tt: int = 0) -> Module:
        """One in-place pass moving register run `tt` of the coal dimension down by `s`.

        The move stays inside the run: `tt` only offsets the coalesced index, so
        no value crosses into a neighbouring run.
        """
        module = Module("shiftBy%u" % s)
        nCoal = lay["numContOutCoal"]
        assert s <= nCoal, (
            "shift=%u must be at most numContOutCoal=%u; the oracle in "
            "tdm_iterate_edge.evaluate() rejects wider shifts" % (s, nCoal)
        )
        nPrep = lay["numOutputsPrep"]
        strideCoal = lay["regStrideCoal"]
        stridePrep = lay["regStridePrep"]
        permOffset = (lay["threadInterval"] % kernel["WavefrontSize"]) * writer.states.bpr

        ttOffset = tt * lay["numRegInMIBCoal"]

        def accIdx(coal, prep):
            return arch2acc[(coal + ttOffset) * strideCoal + prep * stridePrep]

        staging = writer.vgprPool.checkOut(s, tag="unshiftStage%u" % s)
        for p in range(nPrep):
            # The neighbour's low `s` registers are overwritten by this pass, so
            # capture them before any in-lane move runs.
            for k in range(s):
                src = writer.accVgprReadWriteIndex(kernel, accIdx(k, p))
                module.add(DSBPermuteB32(dst=vgpr(staging + k), src0=vgpr(permVgpr),
                                         src1=src, ds=DSModifiers(na=1, offset=permOffset),
                                         comment="neighbour coal[%u] prep[%u]" % (k, p)))
            module.add(SWaitCnt(dscnt=0, comment="wait for the permutes"))
            # Ascending j keeps the in-lane move safe in place.
            for j in range(nCoal - s):
                dst = writer.accVgprReadWriteIndex(kernel, accIdx(j, p))
                src = writer.accVgprReadWriteIndex(kernel, accIdx(j + s, p))
                copyInst = writer.accVgprReadWriteFunction(kernel, accIdx(j, p), False)
                module.add(copyInst(dst=dst, src=src, comment=""))
            for k in range(s):
                dst = writer.accVgprReadWriteIndex(kernel, accIdx(nCoal - s + k, p))
                copyInst = writer.accVgprReadWriteFunction(kernel, accIdx(nCoal - s + k, p), False)
                module.add(copyInst(dst=dst, src=vgpr(staging + k), comment=""))
        writer.vgprPool.checkIn(staging)
        return module
