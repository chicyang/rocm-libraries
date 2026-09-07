# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from Tensile.SolutionStructs.tdm_iterate_edge import coalLayout, evaluate, geometry, shift_steps

pytestmark = pytest.mark.unit


class _FakeDataType:
    """Mirrors the slice of the real DataType API the oracle uses."""

    def __init__(self, nbytes=2.0):
        self._nbytes = nbytes

    def numBytes(self):
        return self._nbytes


def _state(**ovr):
    """Reference gfx1250 bf16 config: MT 256x256, NumWaves 4, VW 8, tile_dim1 8."""
    s = dict(
        NumWaves=4,
        DepthU=128,
        MacroTile0=256,
        MacroTile1=256,
        MatrixInstM=16,
        MatrixInstN=16,
        MatrixInstBM=1,
        MatrixInstBN=1,
        VectorWidthA=8,
        VectorWidthB=8,
        LdsBlockSizePerPadA=2048,
        LdsBlockSizePerPadB=2048,
        TDMSplit=0,
        UseSubtileImpl=0,
        SourceSwap=True,
        WavefrontSize=32,
        MIWaveGroup=[2, 2],
        MIWaveTile=[8, 8],
        MIOutputVectorWidth=8,
        MIArchVgpr=True,
        enableTDMA=1,
        enableTDMB=1,
        _TDMIterateModeA=True,
        _TDMIterateModeB=True,
        ProblemType=dict(Sparse=0, DataTypeA=_FakeDataType(), DataTypeB=_FakeDataType()),
    )
    s["ProblemType"] = {**s["ProblemType"], **ovr.pop("ProblemType", {})}
    s.update(ovr)
    return s


def test_geometry_matches_reference_config():
    g = geometry(_state(), "A")
    assert g["numComp"] == 2
    assert g["rowsPerWave"] == 128
    assert g["bytesPerRow"] == 256
    assert g["tileDim1"] == 8
    assert g["waveBlockSpan"] == 128


def test_reference_config_is_applicable():
    r = evaluate(_state(), "A")
    assert r["applicable"] is True
    assert r["reason"] == ""


def test_not_applicable_when_iterate_mode_off():
    r = evaluate(_state(_TDMIterateModeA=False), "A")
    assert r["applicable"] is False
    assert "iterate" in r["reason"]


def test_not_applicable_for_subtile():
    r = evaluate(_state(UseSubtileImpl=1), "A")
    assert r["applicable"] is False
    assert "subtile" in r["reason"].lower()


def test_not_applicable_when_not_wave_separated():
    r = evaluate(_state(NumWaves=1), "A")
    assert r["applicable"] is False
    assert "wave-separated" in r["reason"]


def test_not_applicable_when_component_spans_two_mi_waves():
    # MIWaveGroup shrinks the per-wave coord span to 64 while rowsPerWave stays
    # 128, so the shifted component would cross an MI wave boundary.
    r = evaluate(_state(VectorWidthA=4), "A")
    assert r["applicable"] is False
    assert "wave block" in r["reason"]


def test_not_applicable_when_tile_dim1_not_power_of_two():
    # 1536 / 256 = 6 rows per pad block.
    r = evaluate(_state(LdsBlockSizePerPadA=1536), "A")
    assert r["applicable"] is False
    assert "power of 2" in r["reason"]


def test_not_applicable_when_rows_per_wave_not_multiple_of_tile_dim1():
    # rowsPerWave = 8 / 2 = 4, still a power of 2 but shorter than tile_dim1 = 8,
    # so a component cannot hold a whole walk step.
    r = evaluate(_state(MacroTile0=8), "A")
    assert r["applicable"] is False
    assert "multiple" in r["reason"]


def test_tdm_split_halves_rows_per_wave():
    g = geometry(_state(TDMSplit=1), "A")
    assert g["rowsPerWave"] == 64


def test_tdm_split_with_sparse_keeps_divisor_one():
    g = geometry(_state(TDMSplit=1, ProblemType={"Sparse": 1}), "A")
    assert g["rowsPerWave"] == 128


def test_tile_dim1_zero_when_lbspp_not_whole_number_of_rows():
    # 2000 / 256 = 7.8125, not a whole number of rows.
    g = geometry(_state(LdsBlockSizePerPadA=2000), "A")
    assert g["tileDim1"] == 0
    r = evaluate(_state(LdsBlockSizePerPadA=2000), "A")
    assert r["applicable"] is False


def test_not_applicable_when_tdm_disabled_for_tensor():
    r = evaluate(_state(enableTDMB=0), "A")
    assert r["applicable"] is False
    assert "wave-separated" in r["reason"]


def test_b_tensor_uses_macro_tile1_and_matrix_inst_n():
    g = geometry(_state(MacroTile1=512), "B")
    assert g["rowsPerWave"] == 256
    assert g["waveBlockSpan"] == 128


def test_shift_steps_for_tile_dim1_8():
    assert shift_steps(8) == [4, 2, 1]


def test_shift_steps_for_tile_dim1_32():
    assert shift_steps(32) == [16, 8, 4, 2, 1]


def test_shift_steps_for_tile_dim1_2():
    assert shift_steps(2) == [1]


from Tensile.SolutionStructs.tdm_iterate_edge import apply_policy


def _policy_state(**ovr):
    s = _state(**ovr)
    s.setdefault("AssertFree0ElementMultiple", 1)
    s.setdefault("AssertFree1ElementMultiple", 1)
    return s


def _collect(msgs):
    return lambda msg: msgs.append(msg)


def test_policy_sets_shift_flag_and_leaves_assert_alone():
    s = _policy_state()
    msgs = []
    assert apply_policy(s, _collect(msgs)) is True
    assert s["_TDMIterEdgeShiftA"] is True
    assert s["_TDMIterEdgeShiftB"] is True
    assert s["AssertFree0ElementMultiple"] == 1
    assert s["AssertFree1ElementMultiple"] == 1
    assert msgs == []


def test_policy_keeps_assert_when_shift_not_applicable():
    # VectorWidthA=4 shrinks the A wave block span; A falls back, B still shifts.
    s = _policy_state(VectorWidthA=4)
    msgs = []
    assert apply_policy(s, _collect(msgs)) is True
    assert s["_TDMIterEdgeShiftA"] is False
    assert s["AssertFree0ElementMultiple"] == 8
    assert s["_TDMIterEdgeShiftB"] is True
    assert s["AssertFree1ElementMultiple"] == 1


def test_policy_preserves_an_existing_larger_multiple():
    s = _policy_state(VectorWidthA=4, AssertFree0ElementMultiple=16)
    apply_policy(s, _collect([]))
    assert s["AssertFree0ElementMultiple"] == 16


def test_policy_rejects_macro_tile_not_multiple_of_tile_dim1():
    # tile_dim1 = 8; MacroTile0 = 132 is not a whole number of steps.
    s = _policy_state(MacroTile0=132)
    msgs = []
    assert apply_policy(s, _collect(msgs)) is False
    assert any("tile_dim1" in m for m in msgs)


def test_policy_ignores_non_iterate_tensors():
    s = _policy_state(_TDMIterateModeA=False, _TDMIterateModeB=False)
    apply_policy(s, _collect([]))
    assert s["_TDMIterEdgeShiftA"] is False
    assert s["AssertFree0ElementMultiple"] == 1


def test_not_applicable_for_non_pow2_bytes_per_element():
    # 6-bit float style type: 0.75 bytes/element, not an integer power of 2.
    r = evaluate(_state(ProblemType={"DataTypeA": _FakeDataType(nbytes=0.75)}), "A")
    assert r["applicable"] is False
    assert "power of 2" in r["reason"]


def test_applicable_for_1_byte_type():
    # A 1-byte type (e.g. fp8) is a legal power-of-2 bpe and must not be
    # rejected for its element size. DepthU 256 keeps bytesPerRow at 256 so
    # tile_dim1 stays 8 and only the bpe question is under test.
    r = evaluate(
        _state(DepthU=256, ProblemType={"DataTypeA": _FakeDataType(nbytes=1)}), "A"
    )
    assert r["applicable"] is True
    assert r["reason"] == ""


def test_1_byte_type_at_depth_u_128_is_accepted_at_the_shift_width_boundary():
    # bytesPerRow 128 makes tile_dim1 16, whose largest shift step (8) equals
    # numContOutCoal (8) for A. That boundary is a valid single-lane rotation
    # (acc[k] = neighbour acc[k], a coord difference of exactly nCoal == s), so
    # this must be accepted -- neither for the shift width nor the element size.
    r = evaluate(_state(ProblemType={"DataTypeA": _FakeDataType(nbytes=1)}), "A")
    assert r["applicable"] is True
    assert r["reason"] == ""


def test_reference_config_applicable_for_both_tensors_after_layout_guards():
    # Guard against over-rejection: the shipped gfx1250 bf16 configuration must
    # survive every accumulator-layout guard, for A and for B.
    for tc in ("A", "B"):
        r = evaluate(_state(), tc)
        assert r["applicable"] is True, "%s rejected: %s" % (tc, r["reason"])
        assert r["reason"] == ""


def test_shift_step_equal_to_num_cont_out_coal_is_accepted():
    # LBSPP 4096 / 256 bytes-per-row = tile_dim1 16, whose largest shift step is
    # 8 -- exactly numContOutCoal for A. The in-lane move range being empty is
    # not a defect: acc[k] = neighbour acc[k] for every k in [0, s), a coord
    # difference of exactly nCoal == s -- a valid pure one-lane rotation.
    r = evaluate(_state(LdsBlockSizePerPadA=4096), "A")
    assert r["applicable"] is True
    assert r["reason"] == ""


def test_not_applicable_when_more_than_one_mi_outer_tile():
    r = evaluate(_state(MIWaveTile=[16, 8]), "A")
    assert r["applicable"] is False
    assert "miOuterTTCoal" in r["reason"]


def test_not_applicable_when_more_than_one_out_block_in_mi():
    # Halving MIOutputVectorWidth halves numContOutCoal for B, which splits the
    # MI output into two blocks along the coalesced dimension.
    r = evaluate(_state(MIOutputVectorWidth=4), "B")
    assert r["applicable"] is False
    assert "OutBlocksInMI" in r["reason"]


def test_not_applicable_when_matrix_inst_b_coal_greater_than_one():
    r = evaluate(_state(MatrixInstBM=2, MacroTile0=512), "A")
    assert r["applicable"] is False
    assert "matrixInstBCoal" in r["reason"]


def test_not_applicable_for_matrix_inst_4():
    r = evaluate(_state(MatrixInstM=4, MacroTile0=64), "A")
    assert r["applicable"] is False
    assert "MatrixInstM/N == 4" in r["reason"]


def test_not_applicable_without_mi_arch_vgpr():
    r = evaluate(_state(MIArchVgpr=False), "A")
    assert r["applicable"] is False
    assert "MIArchVgpr" in r["reason"]


def test_layout_guards_run_after_the_subtile_early_out():
    # Subtile solutions must keep the subtile answer even when their
    # accumulator layout would fail a later guard.
    r = evaluate(_state(UseSubtileImpl=1, MIArchVgpr=False, MIWaveTile=[16, 8]), "A")
    assert r["applicable"] is False
    assert "subtile" in r["reason"].lower()


def test_not_applicable_for_mx_scaled_tensor():
    # MXFP8 at DepthU 256 / VW 8: bpe 1 is a legal power of 2 and tile_dim1
    # stays 8, so only the MX scale pairing is under test. The scale tensor has
    # its own descriptor, which the edge shift never pulls back.
    for tc in ("A", "B"):
        r = evaluate(
            _state(
                DepthU=256,
                ProblemType={
                    "DataTypeA": _FakeDataType(nbytes=1),
                    "DataTypeB": _FakeDataType(nbytes=1),
                    "MXBlock%s" % tc: 32,
                },
            ),
            tc,
        )
        assert r["applicable"] is False
        assert "MX scale" in r["reason"]


def test_mx_block_on_the_other_tensor_does_not_reject():
    # The guard is per tensor: an MX scale on B says nothing about A's pairing.
    r = evaluate(
        _state(
            DepthU=256,
            ProblemType={
                "DataTypeA": _FakeDataType(nbytes=1),
                "DataTypeB": _FakeDataType(nbytes=1),
                "MXBlockB": 32,
            },
        ),
        "A",
    )
    assert r["applicable"] is True
    assert r["reason"] == ""


def test_not_applicable_under_tdm_split():
    # TDMSplit gives each wave component two disjoint row ranges, while the
    # pull-back targets a single component per wave.
    r = evaluate(_state(TDMSplit=1), "A")
    assert r["applicable"] is False
    assert "TDMSplit" in r["reason"]


def test_policy_falls_back_to_assert_multiple_under_tdm_split():
    s = _policy_state(TDMSplit=1)
    msgs = []
    assert apply_policy(s, _collect(msgs)) is True
    assert s["_TDMIterEdgeShiftA"] is False
    assert s["_TDMIterEdgeShiftB"] is False
    assert msgs == []


def test_coal_layout_is_importable_from_the_oracle():
    lay = coalLayout(_state(), isA=True)
    assert lay["numContOutCoal"] == 8
    assert lay["miOuterTTCoal"] == 1
    assert lay["OutBlocksInMI"] == 1
    assert lay["matrixInstBCoal"] == 1


def _f32_vw1_state(**ovr):
    """f32, VectorWidth 1 config: MatrixInstruction [16,16,4,1,1,1,1,2,2].

    numContOutCoal == 1 for A (SourceSwap True, VectorWidthA 1), tile_dim1 == 2,
    so shift_steps(2) == [1] and the largest step exactly equals numContOutCoal
    -- the boundary case the relaxed guard must accept.
    """
    s = _state(
        VectorWidthA=1,
        VectorWidthB=1,
        DepthU=32,
        LdsBlockSizePerPadA=256,
        LdsBlockSizePerPadB=256,
        MacroTile0=32,
        MacroTile1=32,
        MIWaveTile=[1, 1],
        MIWaveGroup=[2, 2],
        NumWaves=4,
        ProblemType=dict(DataTypeA=_FakeDataType(nbytes=4), DataTypeB=_FakeDataType(nbytes=4)),
    )
    s.update(ovr)
    return s


def test_boundary_shift_equal_to_num_cont_out_coal_is_now_accepted():
    # s == numContOutCoal == 1: a pure one-lane rotation (empty in-lane move,
    # top slot index == 0 == k, never negative). Previously rejected by the
    # strict "<" comparison for no correctness reason.
    g = geometry(_f32_vw1_state(), "A")
    assert g["tileDim1"] == 2
    assert shift_steps(g["tileDim1"]) == [1]
    lay = coalLayout(_f32_vw1_state(), isA=True)
    assert lay["numContOutCoal"] == 1

    r = evaluate(_f32_vw1_state(), "A")
    assert r["applicable"] is True, r["reason"]
    assert r["reason"] == ""


def test_shift_step_still_rejected_when_strictly_greater_than_num_cont_out_coal():
    # LdsBlockSizePerPadA 512 -> bytesPerRow 128 -> tile_dim1 4 -> shift_steps
    # [2, 1], largest step 2, which is strictly greater than numContOutCoal 1.
    r = evaluate(_f32_vw1_state(LdsBlockSizePerPadA=512), "A")
    assert r["applicable"] is False
    assert "shift step" in r["reason"]


def _f32_ott2_state(**ovr):
    """f32-like config whose threads hold two register runs along the coal dim.

    f32 caps VectorWidth at 4 // regPerElem, so MIWaveTile 8 gives
    miOuterTTCoal == 2. NumWaves 8 makes numComp 4, which is exactly
    miOuterTTCoal * MIWaveGroup[0], and MacroTile 256 keeps
    rowsPerWave == waveBlockSpan == 64.
    """
    s = _state(
        NumWaves=8,
        DepthU=64,
        MacroTile0=256,
        MacroTile1=256,
        VectorWidthA=4,
        VectorWidthB=4,
        MIWaveTile=[8, 8],
        MIWaveGroup=[2, 2],
        ProblemType=dict(DataTypeA=_FakeDataType(nbytes=4), DataTypeB=_FakeDataType(nbytes=4)),
    )
    s.update(ovr)
    return s


def test_two_outer_tiles_is_applicable():
    for tc in ("A", "B"):
        lay = coalLayout(_f32_ott2_state(), isA=(tc == "A"))
        assert lay["miOuterTTCoal"] == 2
        g = geometry(_f32_ott2_state(), tc)
        assert g["numComp"] == 4
        assert g["rowsPerWave"] == 64 == g["waveBlockSpan"]
        r = evaluate(_f32_ott2_state(), tc)
        assert r["applicable"] is True, "%s rejected: %s" % (tc, r["reason"])
        assert r["reason"] == ""


def test_two_outer_tiles_still_rejects_multiple_out_blocks():
    # Halving MIOutputVectorWidth splits B's MI output into two coal blocks.
    r = evaluate(_f32_ott2_state(MIOutputVectorWidth=4), "B")
    assert r["applicable"] is False
    assert "OutBlocksInMI" in r["reason"]


def test_two_outer_tiles_still_rejects_matrix_inst_b_coal():
    r = evaluate(_f32_ott2_state(MatrixInstBM=2, MacroTile0=512), "A")
    assert r["applicable"] is False
    assert "matrixInstBCoal" in r["reason"]


def test_rejects_when_tt_wave_split_does_not_cover_the_components():
    # miOuterTTCoal * MIWaveGroup[0] must equal numComp, or the (tt, waveG0)
    # decomposition of cOwn would address the wrong block.
    r = evaluate(_f32_ott2_state(NumWaves=4, MacroTile0=128), "A")
    assert r["applicable"] is False
    assert "miOuterTTCoal" in r["reason"]


def test_rejects_non_power_of_two_wave_group_with_two_outer_tiles():
    # cOwn is split with a shift and a mask, so the divisor must be a power of 2.
    # numComp 6 == miOuterTTCoal 2 * MIWaveGroup[0] 3 and rowsPerWave stays 64,
    # so only the power-of-two question is under test.
    r = evaluate(_f32_ott2_state(MIWaveGroup=[3, 2], NumWaves=12, MacroTile0=384), "A")
    assert r["applicable"] is False
    assert "miWaveGroupCoal" in r["reason"]


def _vw1_ott8_state(**ovr):
    """MIWaveTile 8 with an explicit VectorWidth 1: eight register runs.

    VectorWidthA/B are fork parameters the test YAMLs set directly, so this
    combination is common and is not tied to any particular datatype.
    """
    s = _state(
        NumWaves=32,
        DepthU=64,
        MacroTile0=256,
        MacroTile1=256,
        VectorWidthA=1,
        VectorWidthB=1,
        MIWaveTile=[8, 8],
        MIWaveGroup=[2, 2],
        LdsBlockSizePerPadA=512,
        LdsBlockSizePerPadB=512,
        ProblemType=dict(DataTypeA=_FakeDataType(nbytes=4), DataTypeB=_FakeDataType(nbytes=4)),
    )
    s.update(ovr)
    return s


def test_eight_outer_tiles_is_applicable():
    for tc in ("A", "B"):
        lay = coalLayout(_vw1_ott8_state(), isA=(tc == "A"))
        assert lay["miOuterTTCoal"] == 8
        g = geometry(_vw1_ott8_state(), tc)
        assert g["numComp"] == 16
        assert g["rowsPerWave"] == 16 == g["waveBlockSpan"]
        assert shift_steps(g["tileDim1"]) == [1]
        r = evaluate(_vw1_ott8_state(), tc)
        assert r["applicable"] is True, "%s rejected: %s" % (tc, r["reason"])


def test_rejects_mi_wave_tile_not_a_multiple_of_vector_width():
    # VectorWidth 16 above MIWaveTile 8 truncates miOuterTTCoal to 0. MacroTile0
    # 512 keeps rowsPerWave == waveBlockSpan == 256 so only this is under test.
    r = evaluate(_state(VectorWidthA=16, MacroTile0=512), "A")
    assert r["applicable"] is False
    assert "not a multiple of VectorWidthA" in r["reason"]


def test_rejects_mi_wave_tile_with_non_divisible_vector_width():
    # MIWaveTile 6 / VectorWidth 4 truncates to a single register run, which
    # without this guard would look like the already-supported layout while
    # leaving part of the accumulators unmoved.
    s = _state(MIWaveTile=[6, 8], VectorWidthA=4, MacroTile0=128)
    assert coalLayout(s, isA=True)["miOuterTTCoal"] == 1
    g = geometry(s, "A")
    assert g["rowsPerWave"] == 64 == g["waveBlockSpan"]
    r = evaluate(s, "A")
    assert r["applicable"] is False
    assert "not a multiple of VectorWidthA" in r["reason"]


def test_reference_bf16_config_still_accepted_after_relax():
    # The shipped gfx1250 bf16 configuration (tile_dim1 8, numContOutCoal 8 for
    # A) must remain accepted: its largest shift step is 4, well under the
    # boundary, so relaxing "<" to "<=" cannot change its outcome.
    r = evaluate(_state(), "A")
    assert r["applicable"] is True
    assert r["reason"] == ""
