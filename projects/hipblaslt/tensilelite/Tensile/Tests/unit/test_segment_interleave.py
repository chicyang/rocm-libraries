# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from Tensile.SolutionStructs.segment_interleave import evaluate

pytestmark = pytest.mark.unit

def _vw8_state(**ovr):
    s = dict(NumWaves=4, WavefrontSize=32, MacroTile0=256, MacroTile1=256, DepthU=128,
             LdsOffsetA=0, LdsBlockSizePerPadA=2048, LdsBlockSizePerPadB=2048,
             LdsPadA=8, LdsPadB=8, VectorWidthA=8, VectorWidthB=8,
             MatrixInstM=16, MatrixInstN=16, TDMSplit=0, enableTDMA=1, enableTDMB=1,
             ProblemType=dict(TLUA=0, TLUB=0, Sparse=0, DataType="b", MXBlockA=0, MXBlockB=0))
    s["ProblemType"] = {**s["ProblemType"], **ovr.pop("ProblemType", {})}
    s.update(ovr); return s

def test_vw8_applies_with_handedit_values(monkeypatch):
    monkeypatch.delenv("TENSILE_LDS_SEGMENT_INTERLEAVE", raising=False)
    r = evaluate(_vw8_state())
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 65536, "readWaveStride": 32768}

def test_vw4_skips_fine_vw():
    r = evaluate(_vw8_state(VectorWidthA=4))
    assert r["applicable"] is False and "fine VW" in r["reason"]

def test_small_mt_skips_aligned_deferred():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128))
    assert r["applicable"] is False and "aligned" in r["reason"]

def test_non_square_small_skips():
    # MT0=256, MT1=128: coarse VW ok (16*8=128 >= 256//2), but fA+fB=49536 < SEG
    # -> tight cannot move A1 into the next segment -> skip (aligned branch deferred).
    r = evaluate(_vw8_state(MacroTile1=128))
    assert r["applicable"] is False and "aligned" in r["reason"]

def test_off_switch_disables(monkeypatch):
    monkeypatch.setenv("TENSILE_LDS_SEGMENT_INTERLEAVE", "0")
    assert evaluate(_vw8_state())["applicable"] is False

def test_tdmsplit_skips():
    assert evaluate(_vw8_state(TDMSplit=1))["applicable"] is False

def test_tile_major_skips():
    r = evaluate(_vw8_state(ProblemType={"TLUA": 1}))
    assert r["applicable"] is False and ("tile-major" in r["reason"] or "tlu" in r["reason"])
