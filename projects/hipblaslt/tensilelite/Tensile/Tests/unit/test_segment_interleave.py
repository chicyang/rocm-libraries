# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from Tensile.SolutionStructs.segment_interleave import evaluate, aligned_budget_ok, SEG

pytestmark = pytest.mark.unit

class _FakeDataType:
    # Mirrors the real DataType API the oracle uses, without importing rocisa.
    def __init__(self, bf16=True, half=False, f8=False, nbytes=2.0):
        self._bf16 = bf16
        self._half = half
        self._f8 = f8
        self._nbytes = nbytes
    def isBFloat16(self):
        return self._bf16
    def isHalf(self):
        return self._half
    def is8bitFloat(self):
        return self._f8
    def numBytes(self):
        return self._nbytes

def _vw8_state(**ovr):
    # TLUA/TLUB are TOP-LEVEL state keys (not under ProblemType); DataType is an object.
    s = dict(NumWaves=4, WavefrontSize=32, MacroTile0=256, MacroTile1=256, DepthU=128,
             ISA=(12, 5, 0),
             LdsOffsetA=0, LdsBlockSizePerPadA=2048, LdsBlockSizePerPadB=2048,
             LdsPadA=8, LdsPadB=8, VectorWidthA=8, VectorWidthB=8,
             MatrixInstM=16, MatrixInstN=16, TDMSplit=0, enableTDMA=1, enableTDMB=1,
             UnrollMajorLDSA=1, UnrollMajorLDSB=1,
             ProblemType=dict(Sparse=0, DataType=_FakeDataType(), MXBlockA=0, MXBlockB=0))
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

def test_non_gfx1250_skips():
    # SEG=64KiB layout is gfx1250-specific; other ISAs must not apply the interleave.
    for isa in [(9, 4, 2), (9, 5, 0), (12, 0, 0), (11, 0, 0)]:
        r = evaluate(_vw8_state(ISA=isa))
        assert r["applicable"] is False and "gfx1250" in r["reason"]

def test_vwb_fine_skips():
    # B can be fine-VW even when A is coarse (odd WaveTile -> VWB=1). Must skip:
    # GPU-confirmed MT128x224 VWA4_VWB1 gave wrong results before this gate.
    r = evaluate(_vw8_state(VectorWidthB=1))
    assert r["applicable"] is False and "fine VW" in r["reason"]

def test_small_mt_skips_without_pgr2():
    # Small MT is the aligned candidate, but it requires PGR2 double-buffer; with no
    # PrefetchGlobalRead set it skips to baseline (buffering gate).
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_non_square_small_skips_without_pgr2():
    # MT0=256, MT1=128: coarse VW ok (16*8=128 >= 256//2), but fA+fB < SEG -> small MT;
    # no PGR2 -> buffering gate skip.
    r = evaluate(_vw8_state(MacroTile1=128))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_aligned_applies_small_mt():
    # MT128 bf16: dA=dB=64*128*2=16384, pad=(16384//2048)*16=128, fA=fB=16512, sum<SEG.
    # Aligned pushes A1 to the next segment boundary (SEG).
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2))
    assert r["applicable"] is True and r["aligned"] is True
    assert r["offsets"] == {"ldsBaseB": 16512, "writeStrideBytes": SEG, "readWaveStride": SEG // 2}
    # blockSpan is the post-pad span: writeStride(SEG) re-pads to 66048 (blk2048,pad8,bpe2),
    # B1 end = ldsBaseB(16512) + 66048 + fB(16512) = 99072.
    assert r["blockSpan"] == 16512 + 66048 + 16512
    assert "ALIGNED" in r["segmentMap"]

def test_aligned_unequal_pad_no_overlap():
    # padA=16 > padB=8 (same blk=2048): A1 shifts further than B1, so B's base is pushed
    # right by the gap so B1 lands exactly at A1's end (no overlap).
    # dA=dB=16384; fA=16384+(16384//2048)*(16*2)=16640; fB=16384+(16384//2048)*(8*2)=16512
    # pre=SEG; postA=SEG+(SEG//2048)*32=66560; postB=SEG+(SEG//2048)*16=66048; gap=512
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2,
                            LdsPadA=16, LdsPadB=8))
    assert r["applicable"] is True and r["aligned"] is True
    assert r["offsets"]["ldsBaseB"] == 16640 + 512          # fA + gap
    assert r["blockSpan"] == 16640 + 66560 + 16512          # base + fA + max(postA,postB) + fB
    # A1 end == B1 start (adjacent, no overlap):
    postA, fA = 66560, 16640
    a1_end = postA + fA
    b1_start = r["offsets"]["ldsBaseB"] + 66048              # ldsBaseB + postB
    assert a1_end == b1_start

def test_aligned_skips_pgr_not_2():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=1))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_aligned_skips_1ldsbuffer():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, **{"1LDSBuffer": 1}))
    assert r["applicable"] is False and "1LDSBuffer" in r["reason"]

def test_aligned_skips_subtile():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, UseSubtileImpl=1))
    assert r["applicable"] is False and "subtile" in r["reason"]

def test_aligned_skips_dtlplus():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, DtlPlusLdsBuf=1))
    assert r["applicable"] is False and "DtlPlusLdsBuf" in r["reason"]

def test_aligned_budget_ok_fits():
    # blockSpan 98560 -> max -> roundup 131072; 131072*2=262144 <= 327680 (gfx1250 MaxLDS).
    ok, blk = aligned_budget_ok(98560, 2, 0, 327680)
    assert ok is True and blk == 131072

def test_aligned_budget_too_big():
    # MaxLDS=163840 (gfx950): 131072*2=262144 > 163840 -> reject (would force StoreSwapAddr).
    ok, blk = aligned_budget_ok(98560, 2, 0, 163840)
    assert ok is False and blk is None

def test_aligned_budget_numldsblk_not_2():
    ok, blk = aligned_budget_ok(98560, 3, 0, 327680)
    assert ok is False and blk is None

def test_off_switch_disables(monkeypatch):
    monkeypatch.setenv("TENSILE_LDS_SEGMENT_INTERLEAVE", "0")
    assert evaluate(_vw8_state())["applicable"] is False

def test_parameter_off_skips():
    # LDSSegmentInterleave=0 forces baseline (per-solution tuning knob); default (unset)=on.
    r = evaluate(_vw8_state(LDSSegmentInterleave=0))
    assert r["applicable"] is False and "parameter off" in r["reason"]
    assert evaluate(_vw8_state(LDSSegmentInterleave=1))["applicable"] is True

def test_tdmsplit_skips():
    assert evaluate(_vw8_state(TDMSplit=1))["applicable"] is False

def test_tile_major_skips():
    r = evaluate(_vw8_state(UnrollMajorLDSA=0))  # not unrollMajor -> deferred
    assert r["applicable"] is False and ("unrollMajor" in r["reason"] or "tile-major" in r["reason"])

def test_fp16_applies_same_as_bf16():
    # fp16 has the same bpe (2) and the same write/read paths -> identical offsets.
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=True)}))
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 65536, "readWaveStride": 32768}

def test_fp8_applies_with_depthu256():
    # fp8 (bpe=1) with DepthU=256 gives the same 32768-byte chunk as bf16 DepthU=128.
    # dA=dB=(256//2)*256*1=32768 ; pad=(32768//2048)*(8*1)=128 ; footprintA=32896
    r = evaluate(_vw8_state(DepthU=256, ProblemType={"DataType": _FakeDataType(bf16=False, f8=True, nbytes=1)}))
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 32896, "writeStrideBytes": 65536, "readWaveStride": 65536}

def test_fp32_skips():
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=False, nbytes=4)}))
    assert r["applicable"] is False and ("bf16" in r["reason"] or "fp16" in r["reason"] or "fp8" in r["reason"])
