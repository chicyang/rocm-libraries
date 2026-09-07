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


def _mibShapeCoal(k, isA):
    """MIBShapeCoal spelled out the way ShiftVectorComponentsMFMAAllThread does."""
    numThreadInWave = k["WavefrontSize"]
    matrixInstCoal = k["MatrixInstM"] if isA else k["MatrixInstN"]
    matrixInstPrep = k["MatrixInstN"] if isA else k["MatrixInstM"]
    matrixInstBCoal = k["MatrixInstBM"] if isA else k["MatrixInstBN"]
    vectorWidth = k["VectorWidthA"] if isA else k["VectorWidthB"]
    conThInProcDim = bool(k["SourceSwap"]) ^ (not isA)

    numContOutCoal = vectorWidth if conThInProcDim else k["MIOutputVectorWidth"] * vectorWidth
    outBlocksInMI = (
        1
        if conThInProcDim
        else (vectorWidth * matrixInstCoal * matrixInstPrep)
        // numThreadInWave
        // numContOutCoal
    )
    subMBShapeCoal = (
        (matrixInstCoal * vectorWidth)
        if conThInProcDim
        else ((numThreadInWave // matrixInstPrep) * numContOutCoal)
    )
    return subMBShapeCoal * outBlocksInMI * matrixInstBCoal


def test_num_reg_in_mib_coal_matches_shift_vector_components():
    # The register distance between consecutive tt runs. coalLayout builds it
    # from numContOutCoal * OutBlocksInMI * matrixInstBCoal; the reference
    # implementation divides MIBShapeCoal by numThreadInCoal. They must agree.
    for ss in (True, False):
        for isA in (True, False):
            k = _kernel(sourceSwap=ss)
            lay = coalLayout(k, isA=isA)
            assert lay["numRegInMIBCoal"] == _mibShapeCoal(k, isA) // lay["numThreadInCoal"]


def test_num_reg_in_mib_coal_for_both_source_swap_values():
    assert coalLayout(_kernel(), isA=True)["numRegInMIBCoal"] == 8
    assert coalLayout(_kernel(), isA=False)["numRegInMIBCoal"] == 64
    assert coalLayout(_kernel(sourceSwap=False), isA=True)["numRegInMIBCoal"] == 64
    assert coalLayout(_kernel(sourceSwap=False), isA=False)["numRegInMIBCoal"] == 8


def _f32_kernel(**ovr):
    """f32-like layout: VectorWidth 4 with MIWaveTile 8, so miOuterTTCoal == 2."""
    return _kernel(MIWaveTile=[8, 8], VectorWidthA=4, VectorWidthB=4, **ovr)


def test_num_reg_in_mib_coal_with_two_outer_tiles():
    a = coalLayout(_f32_kernel(), isA=True)
    b = coalLayout(_f32_kernel(), isA=False)
    assert a["miOuterTTCoal"] == 2 and a["numRegInMIBCoal"] == 4
    assert b["miOuterTTCoal"] == 2 and b["numRegInMIBCoal"] == 32


def _accIndices(lay):
    """Every accumulator index the un-shift addresses, pre-arch2acc."""
    return [
        (coal + tt * lay["numRegInMIBCoal"]) * lay["regStrideCoal"]
        + prep * lay["regStridePrep"]
        for tt in range(lay["miOuterTTCoal"])
        for coal in range(lay["numContOutCoal"])
        for prep in range(lay["numOutputsPrep"])
    ]


@pytest.mark.parametrize("sourceSwap", [True, False])
@pytest.mark.parametrize("isA", [True, False])
@pytest.mark.parametrize("miWaveTile", [[8, 8], [16, 16]])
def test_acc_index_map_is_a_bijection(sourceSwap, isA, miWaveTile):
    # A missed or duplicated index is exactly how this addressing goes silently
    # wrong, so require the (tt, coal, prep) triple to hit each register once.
    lay = coalLayout(
        _kernel(sourceSwap=sourceSwap, MIWaveTile=miWaveTile, VectorWidthA=4, VectorWidthB=4),
        isA=isA,
    )
    idx = _accIndices(lay)
    assert sorted(idx) == list(range(len(idx)))
    assert len(idx) == lay["miOuterTTCoal"] * lay["numContOutCoal"] * lay["numOutputsPrep"]


def test_consecutive_tt_runs_are_num_reg_in_mib_coal_apart():
    lay = coalLayout(_f32_kernel(), isA=True)
    step = lay["numRegInMIBCoal"] * lay["regStrideCoal"]
    for coal in range(lay["numContOutCoal"]):
        base = coal * lay["regStrideCoal"]
        assert (coal + 1 * lay["numRegInMIBCoal"]) * lay["regStrideCoal"] == base + step


def test_single_outer_tile_layout_is_unchanged():
    # The shipped bf16 configuration: one register run, so tt is always 0 and
    # the index reduces to the pre-existing coal/prep formula.
    for isA in (True, False):
        lay = coalLayout(_kernel(), isA=isA)
        assert lay["miOuterTTCoal"] == 1
        assert _accIndices(lay) == [
            coal * lay["regStrideCoal"] + prep * lay["regStridePrep"]
            for coal in range(lay["numContOutCoal"])
            for prep in range(lay["numOutputsPrep"])
        ]


def test_mi_wave_group_coal_is_reported():
    assert coalLayout(_kernel(), isA=True)["miWaveGroupCoal"] == 2
    assert coalLayout(_kernel(MIWaveGroup=[4, 1]), isA=True)["miWaveGroupCoal"] == 4
    assert coalLayout(_kernel(MIWaveGroup=[4, 1]), isA=False)["miWaveGroupCoal"] == 1
