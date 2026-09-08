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

When several load components share one MI wave block, the passes run with `exec`
narrowed to the lanes from the boundary component's start upwards, which is what
keeps the components below it out of the move.

`delta` is a runtime value and assembly has no dynamic register index, so every
reachable value gets its own straight-line pass, selected by an equality test.
`delta = 0` matches none of them and costs nothing.

A pass that shifts by `s` takes the top `s` values from the neighbouring lane,
which only lands on the right coordinate while `s <= numContOutCoal`. Whether
one pass per value fits therefore depends on `tile_dim1`:

  * `tile_dim1 - 1 <= numContOutCoal` -- one pass per `delta` in
    `[1, tile_dim1)`, so a shift costs a single pass.
  * otherwise -- `delta` is decomposed into powers of two and one pass is
    emitted per bit, which caps the widest pass at `2 ** floor(log2(delta))` and
    costs up to `log2(tile_dim1)` passes.

The choice is per tensor, made from that tensor's own layout, and it is a pure
code-generation decision: the oracle accepts exactly the same configurations
either way.

`tile_dim1` is `LdsBlockSizePerPad / (DepthU * bpe)`. Rounding the pad block up
to 256 bytes can in principle make it exceed `VectorWidth`, but no accepted
solution reaches that: iterate mode requires `LdsBlockSizePerPad > 1024`, which
needs `DepthU * bpe * VectorWidth > 1024` and hence a row of more than 128
bytes, and rounding a row that long has no effect. Enumerating the legal
`(DepthU, bpe, VectorWidth)` under `DepthU >= MatrixInstK` gives
`tile_dim1` in `{2, 3, 4, 6, 8}` and never `tile_dim1 - 1 > numContOutCoal`.
The decomposed path is therefore currently unreachable; it is kept because the
bound rests on two constraints enforced far from here.
"""

from rocisa.code import Label, Module
from rocisa.container import EXEC, ContinuousRegister, DSModifiers, sgpr, vgpr
from rocisa.instruction import (
    DSBPermuteB32,
    SAndB32,
    SAndB64,
    SCBranchSCC0,
    SCmpEQU32,
    SCSelectB32,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SMulI32,
    SSubU32,
    SWaitCnt,
    VAndB32,
    VCmpGEU32,
    VMovB64,
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
        # One pass per reachable delta whenever the widest of them is a shift a
        # single pass can express; otherwise fall back to one pass per bit.
        unrolled = g["tileDim1"] - 1 <= lay["numContOutCoal"]
        steps = (list(range(1, g["tileDim1"])) if unrolled
                 else tdm_iterate_edge.shift_steps(g["tileDim1"]))

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

        # `cOwn` names the load component that holds the boundary. `subPerBlock`
        # of them share one MI block along the coalesced dimension, a thread
        # holds `miOuterTTCoal` register runs, and consecutive runs of one wave
        # are `miWaveGroupCoal` blocks apart, so `cOwn` splits as
        #   sub = cOwn % subPerBlock                        -- which lane range
        #   waveG0 = (cOwn // subPerBlock) % miWaveGroupCoal -- which wave moves
        #   tt = cOwn // (subPerBlock * miWaveGroupCoal)     -- which register run
        # and `(tt * miWaveGroupCoal + waveG0) * subPerBlock + sub` reproduces
        # `cOwn`. The oracle guarantees both divisors are powers of two whenever
        # the corresponding split is needed, so shifts and masks suffice.
        miOuterTTCoal = lay["miOuterTTCoal"]
        subPerBlock = g["subPerBlock"]
        ttSgpr = None
        if miOuterTTCoal > 1:
            ttSgpr = writer.sgprPool.checkOut(1, tag="unshiftTT%s" % tc, preventOverflow=False)
        subSgpr = None
        if subPerBlock > 1:
            subSgpr = writer.sgprPool.checkOut(1, tag="unshiftSub%s" % tc, preventOverflow=False)

        # This wave's block index along the coalesced dimension. Which wave acts
        # is settled by delta, not by exec: waves that do not own the boundary
        # component get delta = 0 and branch past every pass.
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
        if subPerBlock > 1:
            module.add(SAndB32(sgpr(subSgpr), sgpr(ownerSgpr), subPerBlock - 1,
                               "sub = cOwn %% subPerBlock(%u)" % subPerBlock))
            module.add(SLShiftRightB32(
                dst=sgpr(ownerSgpr), shiftHex=(subPerBlock - 1).bit_length(),
                src=sgpr(ownerSgpr),
                comment="cOwn /= subPerBlock(%u)" % subPerBlock))
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

        # A component finer than the MI block occupies the lane range
        # [sub * rowsPerWave / numContOutCoal, numThreadInCoal) of the run: each
        # lane owns `numContOutCoal` consecutive coordinates, so the component
        # start lands on a lane boundary. Narrowing exec to that range leaves the
        # components below `sub` untouched, which is all the passes have to
        # respect -- everything above `sub` maps to coordinates at or past the
        # free size and is discarded by the existing edge mask.
        execSgpr = None
        laneVgpr = None
        startSgpr = None
        numExecSgpr = 1 if kernel["WavefrontSize"] == 32 else 2
        SMovExec = SMovB32 if numExecSgpr == 1 else SMovB64
        SAndExec = SAndB32 if numExecSgpr == 1 else SAndB64
        if subPerBlock > 1:
            laneVgpr = writer.vgprPool.checkOut(1, "unshiftLaneCoal")
            module.add(VAndB32(dst=vgpr(laneVgpr), src0=kernel["WavefrontSize"] - 1,
                               src1=vgpr("Serial"), comment="lane index"))
            if lay["threadInterval"] > 1:
                module.add(vectorStaticDivide(laneVgpr, laneVgpr, lay["threadInterval"],
                                              tmpVgprRes))
            with writer.allocTmpSgpr(writer.states.laneSGPRCount, tag="unshiftLane") as laneTmp:
                module.add(vectorStaticRemainder(dummy, laneVgpr, laneVgpr,
                                                 lay["numThreadInCoal"], tmpVgprRes, laneTmp))
            lanesPerSub = g["rowsPerWave"] // lay["numContOutCoal"]
            startSgpr = writer.sgprPool.checkOut(1, tag="unshiftSubStart%s" % tc,
                                                 preventOverflow=False)
            module.add(SMulI32(sgpr(startSgpr), lanesPerSub, sgpr(subSgpr),
                               "first lane of sub, %u lanes per component" % lanesPerSub))
            execSgpr = writer.sgprPool.checkOutAligned(numExecSgpr, numExecSgpr,
                                                       preventOverflow=False)
            module.add(SMovExec(dst=sgpr(execSgpr, numExecSgpr), src=EXEC(),
                                comment="save EXEC"))
            # v_cmpx lowers through VCC, so the mask is built in a temporary of
            # its own and folded into EXEC by hand; VCC is left alone.
            laneCount = writer.states.laneSGPRCount
            with writer.allocTmpSgpr(laneCount, tag="unshiftLaneMask") as maskInfo:
                module.add(VCmpGEU32(dst=sgpr(maskInfo.idx, laneCount),
                                     src0=vgpr(laneVgpr), src1=sgpr(startSgpr),
                                     comment="lanes at or above sub"))
                module.add(SAndExec(dst=EXEC(), src0=EXEC(),
                                    src1=sgpr(maskInfo.idx, numExecSgpr),
                                    comment="keep only the lanes at or above sub"))

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
                if unrolled:
                    # Exactly one value of delta selects this pass, so the
                    # remaining compares below it cannot also match.
                    module.add(SCmpEQU32(src0=sgpr(deltaSgpr), src1=s,
                                         comment="delta == %u ?" % s))
                else:
                    module.add(SAndB32(sgpr(rowsSgpr), sgpr(deltaSgpr), s,
                                       "delta & %u ?" % s))
                module.add(SCBranchSCC0(labelName=skip.getLabelName(),
                                        comment="skip the shift-by-%u pass" % s))
                module.add(self._shiftBy(writer, kernel, lay, arch2acc, s, permVgpr, tt))
                module.add(skip)
            if ttSkip is not None:
                module.add(ttSkip)

        if execSgpr is not None:
            module.add(SMovExec(dst=EXEC(), src=sgpr(execSgpr, numExecSgpr),
                                comment="restore EXEC"))
            writer.sgprPool.checkIn(execSgpr)
            writer.sgprPool.checkIn(startSgpr)
            writer.vgprPool.checkIn(laneVgpr)
        if subSgpr is not None:
            writer.sgprPool.checkIn(subSgpr)
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

    @staticmethod
    def _prepGroups(kernel, accIdx, nCoal: int, nPrep: int) -> list:
        """Group the prep indices so each group's in-lane moves can be merged.

        Two prep positions can share a `v_mov_b64` when their accumulators are
        an even-aligned consecutive pair at every coalesced position, so the
        partner is looked up through `arch2acc` rather than assumed: the
        permutation puts A's neighbour one prep away and B's one coal position
        away, and only the former is a pairing this emitter can use.

        Returns a list of tuples, each holding one or two prep indices.
        """
        # A 64-bit move addresses a plain register pair; an AGPR operand has no
        # such form, so pairing is only offered on the arch-VGPR layout.
        if not kernel.get("MIArchVgpr", False):
            return [(p,) for p in range(nPrep)]
        byFirst = {}
        for p in range(nPrep):
            byFirst.setdefault(accIdx(0, p), p)
        groups = []
        taken = set()
        for p in range(nPrep):
            if p in taken:
                continue
            taken.add(p)
            p2 = byFirst.get(accIdx(0, p) + 1)
            if (p2 is not None and p2 not in taken
                    and all(accIdx(c, p) % 2 == 0 for c in range(nCoal))
                    and all(accIdx(c, p) + 1 == accIdx(c, p2) for c in range(nCoal))):
                taken.add(p2)
                groups.append((p, p2))
            else:
                groups.append((p,))
        return groups

    def _shiftBy(self, writer, kernel, lay, arch2acc, s: int, permVgpr: int,
                 tt: int = 0) -> Module:
        """One in-place pass moving register run `tt` of the coal dimension down by `s`.

        The move stays inside the run: `tt` only offsets the coalesced index, so
        no value crosses into a neighbouring run.

        Prep positions are processed in the groups `_prepGroups` forms. A paired
        group issues its in-lane moves as `v_mov_b64`, which is safe because the
        two preps address disjoint registers, so a merged move can never read a
        value its partner has already overwritten. The permutes for the whole
        group are issued before any move, since a move overwrites registers a
        later permute in the same group would otherwise read.

        Only the in-lane `acc <- acc` moves merge. The writebacks from staging
        do not: one staging buffer serves every prep in the group, so two preps'
        writebacks share a source register and cannot form a 64-bit pair.

        Groups run one stage deep in software pipeline: two staging buffers are
        held and used alternately, so group `i + 1`'s permutes are issued before
        group `i` moves and the wait ahead of group `i`'s moves only has to
        retire group `i`'s own permutes -- `s_wait_dscnt` names the count group
        `i + 1` is allowed to leave in flight. Distinct groups hold distinct
        prep positions and therefore disjoint accumulators, so an early permute
        never reads a register the group ahead of it is about to overwrite; the
        assertion below states that for the actual `arch2acc` mapping. The
        buffer a permute writes was last read by the group two behind it, whose
        moves have already been emitted, and a VALU operand is read at issue, so
        alternating buffers needs no extra wait.
        """
        module = Module("shiftBy%u" % s)
        nCoal = lay["numContOutCoal"]
        assert s <= nCoal, (
            "shift=%u must be at most numContOutCoal=%u; the emitter falls back "
            "to the power-of-two decomposition above that width" % (s, nCoal)
        )
        nPrep = lay["numOutputsPrep"]
        strideCoal = lay["regStrideCoal"]
        stridePrep = lay["regStridePrep"]
        permOffset = (lay["threadInterval"] % kernel["WavefrontSize"]) * writer.states.bpr

        ttOffset = tt * lay["numRegInMIBCoal"]

        def accIdx(coal, prep):
            return arch2acc[(coal + ttOffset) * strideCoal + prep * stridePrep]

        groups = self._prepGroups(kernel, accIdx, nCoal, nPrep)
        stageWidth = max(len(gp) for gp in groups) * s
        # A permute reads the low `s` coalesced registers of its group; the
        # moves rewrite every coalesced register of theirs. Neighbouring groups
        # must not overlap there, or issuing a group's permutes ahead of the
        # previous group's moves would read values that pass has already moved.
        for i in range(len(groups) - 1):
            reads = {accIdx(k, p) for p in groups[i + 1] for k in range(s)}
            writes = {accIdx(j, p) for p in groups[i] for j in range(nCoal)}
            assert not (reads & writes), (
                "prep groups %s and %s share accumulators %s; the un-shift "
                "pipeline requires neighbouring groups to be disjoint"
                % (groups[i], groups[i + 1], sorted(reads & writes))
            )
        staging = writer.vgprPool.checkOutAligned(2 * stageWidth, 2,
                                                  tag="unshiftStage%u" % s)

        def stageBase(i):
            return staging + (i % 2) * stageWidth

        def issuePermutes(i):
            # The neighbour's low `s` registers are overwritten by this pass, so
            # capture them before any in-lane move of this group runs.
            base = stageBase(i)
            for gi, p in enumerate(groups[i]):
                for k in range(s):
                    src = writer.accVgprReadWriteIndex(kernel, accIdx(k, p))
                    module.add(DSBPermuteB32(
                        dst=vgpr(base + gi * s + k), src0=vgpr(permVgpr),
                        src1=src, ds=DSModifiers(na=1, offset=permOffset),
                        comment="neighbour coal[%u] prep[%u]" % (k, p)))

        def emitMoves(i):
            gp = groups[i]
            base = stageBase(i)
            # Ascending j keeps the in-lane move safe in place.
            for j in range(nCoal - s):
                if len(gp) == 2:
                    module.add(VMovB64(
                        dst=writer.accVgprReadWriteIndex(kernel, accIdx(j, gp[0]), 2),
                        src=writer.accVgprReadWriteIndex(kernel, accIdx(j + s, gp[0]), 2),
                        comment=""))
                    continue
                p = gp[0]
                dst = writer.accVgprReadWriteIndex(kernel, accIdx(j, p))
                src = writer.accVgprReadWriteIndex(kernel, accIdx(j + s, p))
                copyInst = writer.accVgprReadWriteFunction(kernel, accIdx(j, p), False)
                module.add(copyInst(dst=dst, src=src, comment=""))
            for gi, p in enumerate(gp):
                for k in range(s):
                    idx = accIdx(nCoal - s + k, p)
                    copyInst = writer.accVgprReadWriteFunction(kernel, idx, False)
                    module.add(copyInst(dst=writer.accVgprReadWriteIndex(kernel, idx),
                                        src=vgpr(base + gi * s + k), comment=""))

        issuePermutes(0)
        for i in range(len(groups)):
            inFlight = 0
            if i + 1 < len(groups):
                issuePermutes(i + 1)
                inFlight = len(groups[i + 1]) * s
            module.add(SWaitCnt(
                dscnt=inFlight,
                comment="permutes of prep group %u are back (%u still in flight)"
                        % (i, inFlight)))
            emitMoves(i)
        writer.vgprPool.checkIn(staging)
        return module
