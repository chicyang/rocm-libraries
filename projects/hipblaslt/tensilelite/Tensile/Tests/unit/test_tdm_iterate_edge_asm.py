# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Instruction-level checks for the TDM iterate edge shift emitters.

Uses a minimal stub writer, following the harness idiom in
test_streamk_dponly_sgpr_reduction.py: only the surface the emitters touch is
mocked, and assertions are on the emitted Module's instruction stream.
"""

from contextlib import contextmanager

import pytest
from rocisa.code import Module
from rocisa.container import ContinuousRegister

import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


def _TmpSgpr(idx, size):
    """The real allocTmpSgpr hands back a ContinuousRegister; mirror that."""
    return ContinuousRegister(idx, size)


class _StubWriter:
    """Just enough of KernelWriterAssembly for the edge-shift emitters."""

    def __init__(self):
        self._next = 60

    @contextmanager
    def allocTmpSgpr(self, size, alignment=None, tag=""):
        idx = self._next
        self._next += size
        try:
            yield _TmpSgpr(idx, size)
        finally:
            self._next -= size


def _kernel():
    return {
        "MacroTile0": 256,
        "MacroTile1": 256,
        "NumWaves": 4,
        "DepthU": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstBM": 1,
        "MatrixInstBN": 1,
        "VectorWidthA": 8,
        "VectorWidthB": 8,
        "LdsBlockSizePerPadA": 2048,
        "LdsBlockSizePerPadB": 2048,
        "TDMSplit": 0,
        "ProblemType": {"Sparse": 0, "DataTypeA": DataType("H"), "DataTypeB": DataType("H")},
        "_TDMIterEdgeShiftA": True,
        "_TDMIterEdgeShiftB": True,
    }


def _text(mod):
    return str(mod)


def test_delta_uses_and_mask_for_power_of_two_tile_dim1():
    w = _StubWriter()
    mod = Module("t")
    kwa_module.KernelWriterAssembly.tdmIterEdgeDelta(w, mod, _kernel(), "A", 0, 10, 11)
    text = _text(mod)
    # rows clamped to MacroTile0 before the modulo
    assert "s_min_u32" in text
    assert "0x100" in text or " 256" in text
    # (-rows) mod 8 done with a mask, not a divide
    assert "s_and_b32" in text
    assert "s_sub_u32" in text
    assert "s_mul" not in text


def test_owner_uses_shift_for_power_of_two_rows_per_wave():
    w = _StubWriter()
    mod = Module("t")
    kwa_module.KernelWriterAssembly.tdmIterEdgeOwner(w, mod, _kernel(), "A", 0, 10, 12)
    text = _text(mod)
    # rowsPerWave = 128 -> shift right by 7
    assert "s_lshr_b32" in text
    assert "7" in text


def test_b_tensor_owner_uses_macro_tile1():
    w = _StubWriter()
    mod = Module("t")
    k = _kernel()
    k["MacroTile1"] = 512
    kwa_module.KernelWriterAssembly.tdmIterEdgeOwner(w, mod, k, "B", 1, 10, 12)
    # rowsPerWave = 512 / 2 = 256 -> shift right by 8
    assert "8" in _text(mod)


# ---------------------------------------------------------------------------
# Un-shift emitter: register-run dispatch
# ---------------------------------------------------------------------------

import re

from Tensile.Components.TDMIterateUnshift import TDMIterateUnshiftMFMA
from Tensile.KernelWriterModules import accToArchMapper
from Tensile.SolutionStructs.tdm_iterate_edge import coalLayout, geometry, shift_steps


class _Pool:
    """Hands out ascending indices; records that everything is returned."""

    def __init__(self, base):
        self._next = base
        self.live = set()

    def checkOut(self, n, tag="", preventOverflow=False):
        idx = self._next
        self._next += n
        self.live.add(idx)
        return idx

    def checkOutAligned(self, n, alignment, tag="", preventOverflow=False):
        while self._next % alignment:
            self._next += 1
        return self.checkOut(n)

    def checkIn(self, idx):
        self.live.discard(idx)


class _Labels:
    def __init__(self):
        self._n = 0

    def getNameInc(self, name):
        self._n += 1
        return "%s_%u" % (name, self._n)


class _States:
    laneSGPRCount = 1
    bpr = 4
    maxLimitAgprs = 0


class _UnshiftWriter(_StubWriter):
    """Enough of KernelWriterAssembly to run the un-shift emitter end to end."""

    def __init__(self):
        super().__init__()
        self.sgprPool = _Pool(0)
        self.vgprPool = _Pool(0)
        self.labels = _Labels()
        self.states = _States()

    def tdmIterEdgeDelta(self, mod, kernel, tc, ti, rowsSgpr, dstDelta):
        return kwa_module.KernelWriterAssembly.tdmIterEdgeDelta(
            self, mod, kernel, tc, ti, rowsSgpr, dstDelta)

    def tdmIterEdgeOwner(self, mod, kernel, tc, ti, rowsSgpr, dstOwner):
        return kwa_module.KernelWriterAssembly.tdmIterEdgeOwner(
            self, mod, kernel, tc, ti, rowsSgpr, dstOwner)

    def accVgprReadWriteIndex(self, kernel, idx, sz=1):
        return kwa_module.KernelWriterAssembly.accVgprReadWriteIndex(self, kernel, idx, sz)

    def accVgprReadWriteFunction(self, kernel, idx, read=True):
        return kwa_module.KernelWriterAssembly.accVgprReadWriteFunction(self, kernel, idx, read)


def _unshiftKernel(**ovr):
    k = _kernel()
    k.update(
        {
            "MatrixInstBM": 1,
            "MatrixInstBN": 1,
            "MIWaveGroup": [2, 2],
            "MIWaveTile": [8, 8],
            "MIOutputVectorWidth": 8,
            "SourceSwap": True,
            "WavefrontSize": 32,
            "MIArchVgpr": True,
            "UseSubtileImpl": 0,
        }
    )
    k.update(ovr)
    return k


def _f32Kernel(**ovr):
    """miOuterTTCoal == 2."""
    return _unshiftKernel(
        NumWaves=8, DepthU=64, VectorWidthA=4, VectorWidthB=4,
        ProblemType={"Sparse": 0, "DataTypeA": DataType("S"), "DataTypeB": DataType("S")},
        **ovr,
    )


def _vw1Kernel(**ovr):
    """miOuterTTCoal == 8."""
    return _unshiftKernel(
        NumWaves=32, DepthU=64, VectorWidthA=1, VectorWidthB=1,
        LdsBlockSizePerPadA=512, LdsBlockSizePerPadB=512,
        ProblemType={"Sparse": 0, "DataTypeA": DataType("S"), "DataTypeB": DataType("S")},
        **ovr,
    )


def _emit(kernel, tc="A"):
    tP = {"tensorChar": tc, "idx": 0 if tc == "A" else 1, "isA": tc == "A"}
    w = _UnshiftWriter()
    mod = TDMIterateUnshiftMFMA()(w, kernel, tP)
    assert not w.vgprPool.live and not w.sgprPool.live, "emitter leaked a register"
    return _text(mod)


def _branchCount(text):
    return len(re.findall(r"s_cbranch\S*\s+label_TDMIterUnshift", text))


def test_single_register_run_emits_no_extra_branch():
    # The shipped bf16 configuration: one run per thread, so the dispatch adds
    # nothing -- one branch per shift step and no tt compare.
    k = _unshiftKernel()
    assert coalLayout(k, isA=True)["miOuterTTCoal"] == 1
    text = _emit(k)
    assert _branchCount(text) == len(shift_steps(geometry(k, "A")["tileDim1"])) == 3
    assert "_tt0" not in text


def test_two_register_runs_emit_a_flat_compare_and_skip_chain():
    k = _f32Kernel()
    lay = coalLayout(k, isA=True)
    assert lay["miOuterTTCoal"] == 2
    text = _emit(k)
    steps = shift_steps(geometry(k, "A")["tileDim1"])
    # one compare-and-skip per run, plus the per-bit passes inside each run
    assert _branchCount(text) == lay["miOuterTTCoal"] * (len(steps) + 1) == 8
    # Flat: run 0's skip label closes before run 1's compare opens.
    lines = text.splitlines()
    tt0End = next(i for i, l in enumerate(lines) if re.match(r"^label_TDMIterUnshiftA_tt0", l))
    tt1Cmp = next(i for i, l in enumerate(lines) if "register run 1?" in l)
    assert tt0End < tt1Cmp


def test_eight_register_runs_emit_one_block_each():
    k = _vw1Kernel()
    lay = coalLayout(k, isA=True)
    assert lay["miOuterTTCoal"] == 8
    text = _emit(k)
    steps = shift_steps(geometry(k, "A")["tileDim1"])
    assert steps == [1]
    assert _branchCount(text) == lay["miOuterTTCoal"] * (len(steps) + 1) == 16
    for tt in range(8):
        assert "register run %u?" % tt in text


def test_owner_is_split_into_run_and_wave_only_when_needed():
    assert "waveG0" not in _emit(_unshiftKernel())
    text = _emit(_f32Kernel())
    assert "tt = cOwn / miWaveGroupCoal(2)" in text
    assert "waveG0 = cOwn % miWaveGroupCoal(2)" in text


def _valuCPerRun(text, tc):
    """ValuC indices touched inside each register-run block."""
    blocks = re.split(r"^label_TDMIterUnshift%s_tt\d+" % tc, text, flags=re.M)
    return [set(int(m) for m in re.findall(r"vgprValuC\+(\d+)", b)) for b in blocks[:-1]]


@pytest.mark.parametrize("tc", ["A", "B"])
def test_each_register_run_touches_exactly_its_own_registers(tc):
    # The move must never cross between runs: run tt owns
    # (coal + tt*numRegInMIBCoal)*regStrideCoal + prep*regStridePrep.
    k = _f32Kernel()
    lay = coalLayout(k, isA=(tc == "A"))
    _, arch2acc = accToArchMapper(k)
    text = _emit(k, tc)
    runs = _valuCPerRun(text, tc)
    assert len(runs) == lay["miOuterTTCoal"] == 2
    for tt, touched in enumerate(runs):
        expected = {
            arch2acc[(coal + tt * lay["numRegInMIBCoal"]) * lay["regStrideCoal"]
                     + prep * lay["regStridePrep"]]
            for coal in range(lay["numContOutCoal"])
            for prep in range(lay["numOutputsPrep"])
        }
        assert touched == expected
    assert runs[0].isdisjoint(runs[1])


def test_moves_take_their_value_from_delta_rows_above():
    # Direction check on the shift-by-1 pass of run 1: every in-lane move reads
    # the register one coalesced position higher within the same run.
    k = _f32Kernel()
    lay = coalLayout(k, isA=True)
    _, arch2acc = accToArchMapper(k)
    text = _emit(k, "A")
    tt = 1
    off = tt * lay["numRegInMIBCoal"]
    for prep in range(lay["numOutputsPrep"]):
        for j in range(lay["numContOutCoal"] - 1):
            dst = arch2acc[(j + off) * lay["regStrideCoal"] + prep * lay["regStridePrep"]]
            src = arch2acc[(j + 1 + off) * lay["regStrideCoal"] + prep * lay["regStridePrep"]]
            assert "v_mov_b32 v[vgprValuC+%u], v[vgprValuC+%u]" % (dst, src) in text
