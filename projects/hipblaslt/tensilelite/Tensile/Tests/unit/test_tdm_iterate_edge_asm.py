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

import Tensile.KernelWriterAssembly as kwa_module
from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


class _TmpSgpr:
    def __init__(self, idx, size):
        self.idx = idx
        self.size = size


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
