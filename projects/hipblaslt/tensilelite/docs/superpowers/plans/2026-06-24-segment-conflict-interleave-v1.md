# LDS Segment-Conflict Interleave (v1, gfx1250 TDM) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make qualifying gfx1250 wave-separated TDM kernels emit the segment-conflict-free interleaved LDS layout (A0,B0→seg0; A1,B1→seg1) automatically, matching the verified VW8 hand-edit.

**Architecture:** A standalone oracle (`segment_interleave.evaluate(state)`) decides at solution time whether the tight interleave applies and computes the offsets; results go into `state`; three emit sites consume them. Flag-off ⇒ byte-identical to today.

**Tech Stack:** Python (TensileLite generator), pytest unit tests, gfx1250 build via `make co`, validation via the client `run.sh`.

## Global Constraints (verbatim from spec)

- `SEG = 65536` (gfx1250 LDS segment bytes).
- v1 applies only when ALL hold: `isTdmWaveSeparated(kernel)`; per-tensor `unrolledMajor (= not TLU{tc})`; `numComp == NumWaves//2 == 2`; coarse VW (`MI_M_threads*VW >= mt//numComp`); tight branch (`(LdsOffsetA % SEG) + footprintA + footprintB >= SEG`). Else baseline.
- Out of scope (do NOT implement): aligned/small-MacroTile branch, tile-major (tlu), numComp>2, MXS, Sparse, TDMSplit.
- LDS total size, `LdsOffsetA/B`, `blockOffset`, double-buffer swap constants are **NOT** modified (tight = pure reorder).
- `footprint(tc) = dataBytes(tc) + pad(dataBytes(tc))` where `dataBytes = (MacroTile_tc//numComp)*du*bpe` and `pad` uses `LdsBlockSizePerPad{tc}`/`LdsPad{tc}` — i.e. equals the baseline woffset stride.
- Emit units: write woffset multiplier = **pre-pad** `dA+dB`; B-base = **post-pad flat** `LdsOffsetA+footprintA`; read wave-stride = **pre-bpe** `(dA+dB)//bpe`.
- Off-switch env var `TENSILE_LDS_SEGMENT_INTERLEAVE` (default ON); set to `0/false/off` ⇒ feature disabled, output byte-identical to pre-change.
- Verified VW8 reference values: write multiplier `32768→65536`, B-base `66048→33024`, read wave shift `14→15`.

---

### Task 0: Discovery — lock exact edit locations & the read strideWave value

No production code changes. Confirms line numbers (which drift) and resolves the read strideWave ×128 ambiguity empirically.

**Files:**
- Read only: `Tensile/KernelWriterAssembly.py`, `Tensile/Components/LraTileAssignment.py`, `Tensile/SolutionStructs/Solution.py`
- Inspect: a regenerated baseline VW8 `.s` (BBS kernel under `my-custom-build`)

- [ ] **Step 1: Confirm the write woffset site**

Run: `grep -n "woffset = wId \* (mt // numComp" Tensile/KernelWriterAssembly.py`
Expected: one hit inside `initTDMDescriptorWaveSeparatedImpl`, the line `dataBytes = mt // numComp * du * int(bpe * 4) // (4 * dim1Divisor)` immediately above `SMulI32(... dataBytes ...)`. Record the line number as `WOFFSET_LINE`.

- [ ] **Step 2: Confirm the write B-base site**

Run: `grep -n "ldsOffset = woffset + ldsConstOffset" Tensile/KernelWriterAssembly.py`
Expected: hit inside `initTDMDescriptorWaveSeparatedImpl` (`SAddU32(... ldsConstOffset ...)`). Record as `WBASE_LINE`. Confirm `ldsConstOffset = kernel[f"LdsOffset{tc}"]` a few lines above.

- [ ] **Step 3: Confirm the read B-base site**

Run: `grep -n "+= LdsOffset%s (lower)" Tensile/KernelWriterAssembly.py`
Expected: one hit in `lraDeclareAddresses`, `VAddCOU32(... src0=hex(kernel["LdsOffset%s"%tc]) ...)`. Record as `RBASE_LINE`.

- [ ] **Step 4: Confirm the read strideWave site and class**

Run: `sed -n '125,135p;195,201p;236,240p' Tensile/Components/LraTileAssignment.py`
Expected: class `LraTileAssignmentTransposedMFMA` with `DataType("b")` (bf16); `strideWave = numTileInInst * matrixInstT * vectorWidth`; emit `vectorStaticMultiplyAdd(... strideWave ... "W0Stride(%u)" % strideWave)`. Record `STRIDEWAVE_LINE`.

- [ ] **Step 5: Resolve the strideWave numeric value empirically**

Regenerate (or reuse) the baseline VW8 `.s` and read the actual printed value:
Run: `grep -m1 "W0Stride" <path-to-baseline-VW8>.s`
Expected: `... W0Stride(16384) ...` for VW8 (shift 14). Record `BASELINE_STRIDEWAVE = 16384`.
Compute and record: `dA/bpe = (256//2)*128*2 / 2 = 16384`. Confirm `BASELINE_STRIDEWAVE == dA//bpe`. (This proves the read change is `strideWave := (dA+dB)//bpe`, i.e. `×2` for square MT.) If they differ, STOP and re-derive before any edits.

- [ ] **Step 6: Record findings**

Append a short note to the plan file under this task with the four line numbers and the confirmed `strideWave == dA//bpe` relation. No commit (discovery only); proceed.

---

### Task 1: Oracle module `segment_interleave.py`

**Files:**
- Create: `Tensile/SolutionStructs/segment_interleave.py`
- Test: `Tensile/Tests/unit/test_segment_interleave.py`

**Interfaces:**
- Produces: `evaluate(state: dict) -> dict` returning
  `{"applicable": bool, "offsets": {"ldsBaseB": int, "writeStrideBytes": int, "readWaveStride": int}, "reason": str, "segmentMap": str}`.
- Consumes (from `state`): `NumWaves`, `WavefrontSize`, `MacroTile0`, `MacroTile1`, `DepthU`, `LdsOffsetA`, `LdsBlockSizePerPadA/B`, `LdsPadA/B`, `VectorWidthA`, `MatrixInstM/N`, `TDMSplit`, `enableTDMA`, `enableTDMB`, `ProblemType` (`TLUA`,`TLUB`,`Sparse`,`DataType`,`MXBlockA/B`).

- [ ] **Step 1: Write failing tests**

```python
# Tensile/Tests/unit/test_segment_interleave.py
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

def test_non_square_mt_offsets():
    r = evaluate(_vw8_state(MacroTile1=128))  # dA=32768, dB=16384
    assert r["applicable"] is True
    assert r["offsets"]["ldsBaseB"] == 33024            # base + footprintA
    assert r["offsets"]["writeStrideBytes"] == 49152    # dA+dB pre-pad
    assert r["offsets"]["readWaveStride"] == 24576      # (dA+dB)//bpe

def test_off_switch_disables(monkeypatch):
    monkeypatch.setenv("TENSILE_LDS_SEGMENT_INTERLEAVE", "0")
    assert evaluate(_vw8_state())["applicable"] is False

def test_tdmsplit_skips():
    assert evaluate(_vw8_state(TDMSplit=1))["applicable"] is False

def test_tile_major_skips():
    r = evaluate(_vw8_state(ProblemType={"TLUA": 1}))
    assert r["applicable"] is False and ("tile-major" in r["reason"] or "tlu" in r["reason"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd projects/hipblaslt/tensilelite && python -m pytest Tensile/Tests/unit/test_segment_interleave.py -q`
Expected: FAIL with `ModuleNotFoundError: ... segment_interleave`.

- [ ] **Step 3: Implement the oracle**

```python
# Tensile/SolutionStructs/segment_interleave.py
import os

SEG = 65536

_DT_BYTES = {"b": 2, "h": 2, "s": 4}  # v1: bf16("b"); others fall through to skip via DataType check

def _off_switch_disabled():
    return os.environ.get("TENSILE_LDS_SEGMENT_INTERLEAVE", "1").lower() in ("0", "false", "off")

def _bpe(state):
    return _DT_BYTES.get(state["ProblemType"]["DataType"], 2)

def _pad(x, blk, padElems, bpe):
    if blk == 0 or padElems == 0:
        return 0
    return (x // blk) * (padElems * bpe)

def _data_bytes(state, tc):
    numComp = state["NumWaves"] // 2
    mt = state["MacroTile0"] if tc == "A" else state["MacroTile1"]
    return (mt // numComp) * state["DepthU"] * _bpe(state)

def _footprint(state, tc):
    d = _data_bytes(state, tc)
    blk = state["LdsBlockSizePerPad%s" % tc]
    padElems = state["LdsPad%s" % tc]
    return d + _pad(d, blk, padElems, _bpe(state))

def _coarse_vw(state):
    # each port reads one contiguous M-group: MI_M_threads * VW >= mt//numComp
    numComp = state["NumWaves"] // 2
    mi_m_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    return mi_m_threads * state["VectorWidthA"] >= state["MacroTile0"] // numComp

def _no(reason):
    return {"applicable": False, "offsets": None, "reason": reason, "segmentMap": ""}

def evaluate(state):
    pt = state["ProblemType"]
    if _off_switch_disabled():                                  return _no("off-switch")
    if not (state.get("enableTDMA") and state.get("enableTDMB") and state["NumWaves"] > 1):
        return _no("not wave-separated TDM")
    if pt.get("TLUA") or pt.get("TLUB"):                        return _no("tile-major (tlu) deferred")
    if state["NumWaves"] // 2 != 2:                             return _no("numComp!=2")
    if state.get("TDMSplit") or pt.get("MXBlockA") or pt.get("MXBlockB") or pt.get("Sparse"):
        return _no("split/mxs/sparse")
    if pt["DataType"] not in ("b",):                            return _no("v1: bf16 only")
    if not _coarse_vw(state):                                   return _no("fine VW")

    fA, fB = _footprint(state, "A"), _footprint(state, "B")
    dA, dB = _data_bytes(state, "A"), _data_bytes(state, "B")
    base = state["LdsOffsetA"]
    if (base % SEG) + fA + fB < SEG:
        return _no("small MacroTile (aligned branch deferred to phase 2)")

    bpe = _bpe(state)
    offsets = {
        "ldsBaseB":         base + fA,          # post-pad flat
        "writeStrideBytes": dA + dB,            # pre-pad
        "readWaveStride":   (dA + dB) // bpe,   # pre-bpe
    }
    a0 = base // SEG
    a1 = (base + fA + fB) // SEG
    seg_map = "CLEAN seg%d={A0,B0} seg%d={A1,B1}" % (a0, a1) if a0 != a1 else "PARTIAL"
    return {"applicable": True, "offsets": offsets, "reason": "tight", "segmentMap": seg_map}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tensile/Tests/unit/test_segment_interleave.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add Tensile/SolutionStructs/segment_interleave.py Tensile/Tests/unit/test_segment_interleave.py
git commit -m "feat(tdm): segment-conflict interleave oracle (v1, bf16 tight)"
```

---

### Task 2: Wire oracle into solution state + per-kernel log

**Files:**
- Modify: `Tensile/SolutionStructs/Solution.py` (in `setLdsOffsets` / right after `LdsOffsetB` is finalized)
- Test: `Tensile/Tests/unit/test_segment_interleave_state.py`

**Interfaces:**
- Consumes: `evaluate()` from Task 1.
- Produces: `state["LDSSegInterleave"]: bool`, `state["LDSSegInterleaveOffsets"]: dict|None`.

- [ ] **Step 1: Write failing test**

```python
# Tensile/Tests/unit/test_segment_interleave_state.py
import pytest
pytestmark = pytest.mark.unit

def test_state_keys_present_after_eval(monkeypatch):
    monkeypatch.delenv("TENSILE_LDS_SEGMENT_INTERLEAVE", raising=False)
    from Tensile.SolutionStructs.segment_interleave import evaluate
    from Tensile.Tests.unit.test_segment_interleave import _vw8_state
    s = _vw8_state()
    res = evaluate(s)
    s["LDSSegInterleave"] = res["applicable"]
    s["LDSSegInterleaveOffsets"] = res["offsets"]
    assert s["LDSSegInterleave"] is True
    assert s["LDSSegInterleaveOffsets"]["ldsBaseB"] == 33024
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest Tensile/Tests/unit/test_segment_interleave_state.py -q`
Expected: PASS already for the standalone helper test (it exercises evaluate()); this guards the key names the emit sites use. If import path wrong, FAIL — fix import. (This task's real change is the Solution.py wiring below; the test pins the contract.)

- [ ] **Step 3: Wire into `setLdsOffsets`**

In `Tensile/SolutionStructs/Solution.py`, immediately after `state["LdsOffsetB"]` is assigned in `setLdsOffsets` (locate via `grep -n 'state\["LdsOffsetB"\] = rawLdsOffsetB' Tensile/SolutionStructs/Solution.py`), add:

```python
    from Tensile.SolutionStructs.segment_interleave import evaluate as _segIntEval
    _segRes = _segIntEval(state)
    state["LDSSegInterleave"] = _segRes["applicable"]
    state["LDSSegInterleaveOffsets"] = _segRes["offsets"]
    if _segRes["applicable"]:
        print("[LDSSegInterleave] %s: APPLIED %s offsets=%s"
              % (state.get("KernelName", "?"), _segRes["segmentMap"], _segRes["offsets"]))
    else:
        print("[LDSSegInterleave] %s: SKIP reason=%s"
              % (state.get("KernelName", "?"), _segRes["reason"]))
```

(If `state` lacks `LdsOffsetB` at that point for some solution shapes, guard with `if "LdsOffsetB" in state:`.)

- [ ] **Step 4: Run unit suite**

Run: `python -m pytest Tensile/Tests/unit/test_segment_interleave.py Tensile/Tests/unit/test_segment_interleave_state.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Tensile/SolutionStructs/Solution.py Tensile/Tests/unit/test_segment_interleave_state.py
git commit -m "feat(tdm): compute segment-interleave decision in setLdsOffsets + log"
```

---

### Task 3: Emit site 1 — write woffset stride

**Files:**
- Modify: `Tensile/KernelWriterAssembly.py` (`initTDMDescriptorWaveSeparatedImpl`, `WOFFSET_LINE` from Task 0)

**Interfaces:**
- Consumes: `kernel["LDSSegInterleave"]`, `kernel["LDSSegInterleaveOffsets"]["writeStrideBytes"]`.

- [ ] **Step 1: Apply the gated change**

Replace the `dataBytes` used in the woffset multiply so that, when the flag is set, the multiplier becomes `writeStrideBytes` (pre-pad; the existing `+= padBytes` then yields the footprint sum):

```python
      dataBytes = mt // numComp * du * int(bpe * 4) // (4 * dim1Divisor)
      if kernel.get("LDSSegInterleave"):
          dataBytes = kernel["LDSSegInterleaveOffsets"]["writeStrideBytes"]
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), dataBytes, ...))
```

(Keep the existing comment string; the surrounding pad-add lines are unchanged.)

- [ ] **Step 2: Sanity-build the generator (import + lint)**

Run: `python -c "import Tensile.KernelWriterAssembly"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add Tensile/KernelWriterAssembly.py
git commit -m "feat(tdm): interleave write woffset stride (emit site 1)"
```

---

### Task 4: Emit site 2 — write & read B base

**Files:**
- Modify: `Tensile/KernelWriterAssembly.py` (`WBASE_LINE` write; `RBASE_LINE` read in `lraDeclareAddresses`)

**Interfaces:**
- Consumes: `kernel["LDSSegInterleave"]`, `kernel["LDSSegInterleaveOffsets"]["ldsBaseB"]`.

- [ ] **Step 1: Write-side B base (WBASE_LINE)**

```python
      ldsConstOffset = kernel[f"LdsOffset{tc}"]
      if kernel.get("LDSSegInterleave") and tc == "B":
          ldsConstOffset = kernel["LDSSegInterleaveOffsets"]["ldsBaseB"]
      mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), ldsConstOffset, "ldsOffset = woffset + ldsConstOffset"))
```

- [ ] **Step 2: Read-side B base (`lraDeclareAddresses`, RBASE_LINE)**

```python
    elif (kernel["LdsOffset%s"%tc] != 0) or (kernel.get("LDSSegInterleave") and tc == "B"):
      _bbase = kernel["LDSSegInterleaveOffsets"]["ldsBaseB"] if (kernel.get("LDSSegInterleave") and tc == "B") \
               else kernel["LdsOffset%s"%tc]
      module.add(VAddCOU32(dst=vgpr("LocalReadAddr%s+0"%tc), dst1=VCC(),
                           src0=hex(_bbase), src1=vgpr("LocalReadAddr%s+0"%tc),
                           comment=" += LdsOffset%s (lower)"%tc))
```

- [ ] **Step 3: Import sanity**

Run: `python -c "import Tensile.KernelWriterAssembly"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add Tensile/KernelWriterAssembly.py
git commit -m "feat(tdm): interleave write/read B base (emit site 2)"
```

---

### Task 5: Emit site 3 — read wave stride

**Files:**
- Modify: `Tensile/Components/LraTileAssignment.py` (`LraTileAssignmentTransposedMFMA`, `STRIDEWAVE_LINE`)

**Interfaces:**
- Consumes: `kernel["LDSSegInterleave"]`, `kernel["LDSSegInterleaveOffsets"]["readWaveStride"]`.

- [ ] **Step 1: Apply the gated change**

```python
        strideWave   = numTileInInst * matrixInstT * vectorWidth
        if kernel.get("LDSSegInterleave"):
            strideWave = kernel["LDSSegInterleaveOffsets"]["readWaveStride"]
```

(The `vectorStaticMultiplyAdd(... strideWave ...)` below is unchanged; it emits `lshl` for power-of-two and a multiply otherwise.)

- [ ] **Step 2: Import sanity**

Run: `python -c "import Tensile.Components.LraTileAssignment"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add Tensile/Components/LraTileAssignment.py
git commit -m "feat(tdm): interleave read wave stride (emit site 3)"
```

---

### Task 6: End-to-end verification on gfx1250

**Files:** none (verification only)

- [ ] **Step 1: Regenerate the BBS (VW8) kernel through the pipeline**

Use the project's library-generation entry (confirm exact command from `my-custom-build/TensileCreateLibrary.sh`). Capture the regenerated `.s` path.
Expected: `[LDSSegInterleave] <BBS kernel>: APPLIED ...` appears in generation log.

- [ ] **Step 2: Inspect the generated `.s` matches the hand-edit**

Run: `grep -nE "s_mul_i32 s.., s.., 65536|0x8100|v_lshl_add_u32 v., v., 15," <regenerated>.s | head`
Expected: write multiplier `65536`, read shift `15`, B base `0x8100` present.

- [ ] **Step 3: Build and validate**

Run: `make co TENSILE_OUT=my-custom-build TARGET_ARCH=gfx1250 ARCH=gfx1250 WAVE=32`
then `bash ./my-custom-build/1_BenchmarkProblems/Cijk_Alik_Bljk_BBS_BH_UserArgs_00/00_Final/build/run.sh 2>&1 | grep -oE "(PASSED|FAILED),"`
Expected: `PASSED,`.

- [ ] **Step 4: Inert-when-off regression**

Regenerate the same kernel with `TENSILE_LDS_SEGMENT_INTERLEAVE=0`; diff its `.s` against a pre-change baseline `.s`.
Expected: empty diff (feature off ⇒ byte-identical).

- [ ] **Step 5: Non-qualifying regression**

Regenerate a VW4 kernel; confirm log shows `SKIP reason="fine VW"` and its `.s` is unchanged vs pre-change.
Expected: SKIP logged, empty diff.

- [ ] **Step 6: Commit verification notes**

```bash
git commit --allow-empty -m "test(tdm): e2e segment-interleave VW8 PASSED + inert-when-off verified"
```

---

## Self-Review

- **Spec coverage:** scope/guards (Task 1), oracle two-unit-correctness + units (Task 1), state wiring + log (Task 2), 3 emit sites (Tasks 3–5), off-switch (Task 1 oracle + tests), verification incl. inert-when-off + non-qualifying (Task 6). Aligned/tlu explicitly deferred (not tasked). ✓
- **Placeholder scan:** Task 0 leaves line numbers as named tokens (`WOFFSET_LINE` etc.) intentionally — they are resolved empirically in Task 0 before use; all code steps show concrete code. No TBD/"add error handling".
- **Type consistency:** `evaluate()` returns `{"applicable","offsets","reason","segmentMap"}`; offsets keys `ldsBaseB/writeStrideBytes/readWaveStride` used identically in Tasks 3–5 and tests. `state` keys `LDSSegInterleave`/`LDSSegInterleaveOffsets` consistent across Tasks 2–5.
- **Open item:** Task 6 Step 1 exact regen command — to be read from `my-custom-build/TensileCreateLibrary.sh` at execution time (noted in spec §7).
