# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout maths for the iterate-mode accumulator un-shift.

The register-index maths is the part that goes wrong silently, so it is
extracted into pure helpers and tested directly; the emitted instruction stream
is checked end-to-end in Task 5 against real generated assembly.
"""

import pytest

from Tensile.SolutionStructs.tdm_iterate_edge import coalLayout

pytestmark = pytest.mark.unit


def _kernel(sourceSwap=True, **ovr):
    k = {
        "SourceSwap": sourceSwap,
        "WavefrontSize": 32,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstBM": 1,
        "MatrixInstBN": 1,
        "MIWaveGroup": [2, 2],
        "MIWaveTile": [8, 8],
        "MIOutputVectorWidth": 8,
        "VectorWidthA": 8,
        "VectorWidthB": 8,
    }
    k.update(ovr)
    return k


def test_source_swap_true_matrix_a_is_lane_spread():
    lay = coalLayout(_kernel(), isA=True)
    assert lay["numContOutCoal"] == 8
    assert lay["numThreadInCoal"] == 16
    assert lay["threadInterval"] == 1
    assert lay["regStrideCoal"] == 1
    assert lay["regStridePrep"] == 8
    assert lay["numOutputsPrep"] == 64


def test_source_swap_true_matrix_b_is_lane_contiguous():
    lay = coalLayout(_kernel(), isA=False)
    assert lay["numContOutCoal"] == 64
    assert lay["numThreadInCoal"] == 2
    assert lay["threadInterval"] == 16
    assert lay["regStrideCoal"] == 8
    assert lay["regStridePrep"] == 1
    assert lay["numOutputsPrep"] == 8


def test_source_swap_false_mirrors_a_and_b():
    a = coalLayout(_kernel(sourceSwap=False), isA=True)
    b = coalLayout(_kernel(sourceSwap=False), isA=False)
    assert a["numContOutCoal"] == 64 and a["threadInterval"] == 16
    assert b["numContOutCoal"] == 8 and b["threadInterval"] == 1


def test_total_registers_is_layout_invariant():
    for ss in (True, False):
        for isA in (True, False):
            lay = coalLayout(_kernel(sourceSwap=ss), isA=isA)
            assert lay["numContOutCoal"] * lay["numOutputsPrep"] == 512
