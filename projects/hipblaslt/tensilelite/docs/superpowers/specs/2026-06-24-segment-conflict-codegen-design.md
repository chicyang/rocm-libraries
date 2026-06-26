# Design: LDS segment-conflict interleave in codegen (gfx1250 TDM)

**Date:** 2026-06-24
**Status:** Approved (design); pending spec review → implementation plan
**Component:** TensileLite kernel generator (`projects/hipblaslt/tensilelite`)

## 1. Problem

On gfx1250, LDS is 5×64KiB **segments** (`SEG = 65536`). The 4 SIMDs form **2 pairs**, each served by one **port** (256 B/cycle; 512 B/cycle peak only when the two ports hit *different* segments in a cycle). Two ports hitting the same segment in the same cycle = a **segment conflict** (throughput halves).

For the gfx1250 **wave-separated** TDM (`tensor_load_to_lds`) path the MFMA read ports are:
- `port0 = {W0, W2}` (even SIMDs, M-group 0) → reads A-half **A0**
- `port1 = {W1, W3}` (odd SIMDs, M-group 1) → reads A-half **A1**

Baseline LDS places both A-halves in one segment (A0 and A1 contiguous):
```
baseline:  seg0 = {A0, A1}   seg1 = {B0, B1}
A-read:  port0 reads A0(seg0), port1 reads A1(seg0)  → same segment → CONFLICT
```

The fix interleaves B between the A-halves so A1 moves into a different segment:
```
interleave: seg0 = {A0, B0}   seg1 = {A1, B1}
A-read:  port0 reads A0(seg0), port1 reads A1(seg1)  → different → no conflict
```
Verified by hand-edit on the VW8 kernel (compiled + validation PASSED on gfx1250): write wave-half stride `32768→65536`, write+read B-base `66048→33024`, read wave-half stride shift `14→15` (`16384→32768`).

### Key property: interleave is monotone — never worse than baseline
Inserting B between A0 and A1 increases the port-to-port distance for the A-read from `< SEG` (baseline: A0/A1 same segment, ~100% of concurrent ds_load pairs collide) to `≥ SEG` (A1 pushed ≥1 segment away). The A-read conflict count can only **decrease or stay equal**. B was already maximally collided in baseline; spreading it cannot make it worse. The shift amounts are multiples of 128 B → **bank-neutral**. The v1 (tight) layout is a **pure reorder within the same allocated span** → total LDS unchanged → no occupancy cost. ⇒ We need only ensure **not worse**, not prove **zero conflict**.

## 2. Scope (v1) and decisions

**v1 applies only when ALL hold** (else baseline, byte-identical):
- gfx1250 **wave-separated** TDM: `isTdmWaveSeparated(kernel)` (`KWA:352` = `enableTDMA & enableTDMB & NumWaves>1`). The plain `initTDMDescriptor` path (all-waves, `mt//numWaves`) has no even/odd port split → out of scope.
- **UnrollMajor on the relevant tensor(s)**: `unrolledMajor = not TLU{tc}` (`KWA:18573/18725`). `TLUA`/`TLUB` are per-tensor — require unrollMajor for the tensors we touch. tile-major (tlu) deferred.
- `numComp == NumWaves//2 == 2` (2 ports ↔ 2 segments).
- **coarse VW**: `MI_M_threads * VW >= mt//numComp` (each port reads a single contiguous M-group). Fine VW (VW4) reads interleaved stripes spanning both M-groups → needs M-split writes that don't exist → out of scope.
- **tight branch only**: `(LdsOffsetA mod SEG) + footprintA + footprintB >= SEG` (A1 lands in a different segment without growing LDS). Small-MacroTile (aligned) case → **deferred to phase 2** (it grows LDS and must touch the size accumulator + blockOffset + double-buffer swap).

| Decision | Choice |
|---|---|
| Activation | **Automatic** when the v1 scope holds |
| Decision logic | **Standalone oracle module**, called at solution time; result (flag + offsets + segment map) stored in `state` |
| LDS size | **Unchanged** (tight = pure reorder); no `Solution.py` size / blockOffset / swap changes in v1 |
| Safety net | **Env-var off-switch** (default ON) + **per-kernel log** (CLEAN / PARTIAL / SKIP+reason) |
| Verification | Oracle unit tests + end-to-end regen→build→run + inert-when-off regression |

### Deferred (later phases, explicitly not v1)
- **Aligned branch** (small MacroTile, `... < SEG`): force A1 to `ceil_to_SEG(LdsOffsetA + footprintA + footprintB)`. Grows LDS → must also update LDS total, `blockOffset`, and the double-buffer swap constants (write `±blockOffset`, read XOR), and re-check the 5×SEG budget + occupancy. Separate phase.
- **tile-major (tlu)** kernels.

## 3. Architecture

One decision, three consumers (single source of truth → write and read cannot diverge, the bug that broke the first hand-edit):
```
solution time (Solution.py setLdsOffsets):
    res = segment_interleave.evaluate(state)
    state["LDSSegInterleave"]        = res.applicable
    state["LDSSegInterleaveOffsets"] = res.offsets    # {strideA, strideB, ldsBaseB, readWaveStride}
    log(res)

kernel emit time — each site branches on the flag and uses the stored offsets:
    1) KWA write woffset stride   2) KWA write/read B-base   3) LraTileAssignment read wave-stride
```
Flag False ⇒ every emit site byte-identical to today.

## 4. The oracle (`Tensile/.../segment_interleave.py`)

Pure function of `state`; no side effects; unit-testable.

```python
SEG = 65536

def footprint(state, tc):
    # authoritative per-wId-half LDS span = the baseline woffset stride (KWA:18791-18800)
    dataBytes = (mt(tc) // numComp) * du * bpe          # mt = MacroTile0 for A, MacroTile1 for B
    return dataBytes + pad(dataBytes, ldsBlockSizePerPad[tc], ldsPadSize[tc])

def evaluate(state) -> Result:
    if off_switch_disabled():                                   return skip("off-switch")
    if not is_tdm_wave_separated(state):                        return skip("not wave-separated TDM")
    if not unrolled_major(state):                               return skip("tile-major (tlu) deferred")
    if state["NumWaves"] // 2 != 2:                             return skip("numComp!=2")
    if state["TDMSplit"] or is_mxs(state) or sparse(state):     return skip("split/mxs/sparse")
    if not coarse_vw(state):                                    return skip("fine VW")

    fA, fB = footprint(state, "A"), footprint(state, "B")       # may differ (non-square MT)
    base   = state["LdsOffsetA"]                                # may be nonzero (half-bank shift; 0 for bf16)

    # v1 = tight branch only
    if (base % SEG) + fA + fB < SEG:
        return skip("small MacroTile (aligned branch deferred to phase 2)")

    dA, dB = dataBytes(state,"A"), dataBytes(state,"B")          # PRE-pad data per half
    offsets = dict(                                              # each consumer gets ITS OWN units:
        ldsBaseB         = base + fA,        # POST-pad flat byte add (write ldsConstOffset & read += LdsOffset)
        writeStrideBytes = dA + dB,          # PRE-pad bytes; write woffset code re-adds pad -> footprint sum
        readWaveStride   = (dA + dB)//bpe,   # PRE-bpe elems; read applies *bpe + pad (VW8: 32768 = shift 15)
    )
    return apply(offsets, segment_map(state, offsets))         # CLEAN or PARTIAL (log only)

# UNITS (critical): branch/segment math uses POST-pad footprints (fA,fB, base%SEG).
# Emit strides are PRE-pad (write) / PRE-bpe (read) because those sites re-apply pad/bpe.
# Assumes dataBytes are multiples of ldsBlockSizePerPad so pad(dA+dB)=padA+padB (verify in oracle).
```

Rules:
- **`footprint = dataBytes + pad(dataBytes)`** is the *real* per-half LDS span — it equals the baseline woffset stride, so it already accounts for the chunk's internal (iterate-mode) padding; do not re-derive a separate "data + pad" guess.
- **chunkA ≠ chunkB** handled (`fA`, `fB` separate). **A0 not assumed at seg0**: branch uses `base % SEG`; `base = LdsOffsetA`.
- **Tight = pure reorder, same total LDS** → applied whenever the scope holds; the monotone property guarantees not-worse, so **no zero-conflict gate** (no EPS, no straddle gate).
- **`segment_map`** computes the padding-accurate per-buffer placement (`base + buf*blockOffset + woffset + pad`) only to label the log `CLEAN` (A fully separated every buffer) vs `PARTIAL` (some A straddle/overlap, still ≤ baseline). Not a gate.
- Emit values for VW8 equal the verified hand-edit (`strides=66048`, `ldsBaseB=33024`, read shift 15).

## 5. Emit-site changes (3 sites, flag-gated, consume stored offsets)

1. **Write woffset stride** — `KernelWriterAssembly.py:18791` (`initTDMDescriptorWaveSeparatedImpl`): replace the `dataBytes` multiplier with `offsets["writeStrideBytes"]` (PRE-pad; the existing `+= padBytes` at 18792-18800 then yields the post-pad footprint). VW8: `32768→65536`.
2. **Write/read B base** — write `ldsConstOffset` (`KWA:18800`, added *after* pad) and read `+= LdsOffset{tc}` (`KWA:5899`) use `offsets["ldsBaseB"]` (post-pad flat) instead of `kernel["LdsOffsetB"]`. VW8: `66048→33024`.
3. **Read wave-stride** — the `LraTileAssignment` subclass from `Component.LraTileAssignment.find()` for the wave-separated/unrollMajor/TDM read (emits `W0Stride(...)`): use `offsets["readWaveStride"]` (PRE-bpe; read re-applies ×bpe + pad) instead of `strideWave`. VW8: `16384→32768` (shift 14→15).

`kernel["LdsOffsetA/B"]` and the LDS-size accumulator are **not** modified (tight = same span). Emit sites are dumb consumers of the stored offsets.

## 6. Safety net

**Off-switch** (env var, default ON), checked first in `evaluate()`:
`TENSILE_LDS_SEGMENT_INTERLEAVE in {0,false,off}` → `evaluate()` returns `applicable=False` for all kernels → byte-identical to pre-change codegen.

**Per-kernel log** (one line at solution time):
```
[LDSSegInterleave] <kernel>: CLEAN   tight  seg0={A0@0,B0@33024} seg1={A1@66048,B1@99072}
[LDSSegInterleave] <kernel>: PARTIAL tight  A1 straddles in buf1 (still <= baseline)
[LDSSegInterleave] <kernel>: SKIP    reason="small MacroTile (aligned deferred)"
[LDSSegInterleave] <kernel>: SKIP    reason="fine VW"
```

## 7. Verification

**Unit (no GPU, `Tensile/Tests/unit`):** synthetic `state` → assert `evaluate()`:
- VW8 MT256×256×128 bf16 (tight) → APPLIED, offsets `{writeStrideBytes=65536, ldsBaseB=33024, readWaveStride=32768}` (== verified hand-edit), log CLEAN.
- Non-square MT (MT0≠MT1) → APPLIED with `fA≠fB`, `ldsBaseB=base+fA`, `writeStrideBytes=dA+dB`, branch using post-pad `fA+fB`.
- Small MT (`(base%SEG)+fA+fB < SEG`) → SKIP "aligned deferred".
- Not wave-separated → SKIP; tile-major (tlu) → SKIP; numComp≠2 → SKIP; VW4 → SKIP "fine VW"; TDMSplit/MXS/Sparse → SKIP; off-switch → SKIP all.
- `LdsOffsetA != 0` case → branch uses `base % SEG`, offsets shifted correctly.

**End-to-end (GPU loop):**
1. Regenerate the BBS (VW8) kernel; inspect `.s` for write woffset multiplier `65536` (→ 66048 after the pad-add), B base `0x8100` (33024), read wave shift `15` — byte-equivalent to the hand-edit.
2. `make co TENSILE_OUT=my-custom-build TARGET_ARCH=gfx1250 ARCH=gfx1250 WAVE=32` → `run.sh` → validation **PASSED**.
3. Regression: a non-qualifying kernel (VW4) generated `.s` diff vs pre-change = empty (flag-off path inert).
4. (Optional) perf compare baseline vs interleave.

## 8. Risks / plan-level TODOs

- **Read-side `find()` subclass** — identify the exact `LraTileAssignment` class for wave-separated/unrollMajor/TDM and confirm its `wtid0` (port) mapping matches the write even/odd; only change that class's `strideWave`. (`Tensile/Components/LraTileAssignment.py`, 5 candidate sites.)
- **`readWaveStride` encoding** — value is PRE-bpe `(dA+dB)/bpe` (VW8: 32768, a power of two → emits as the `lshl` shift 15). If a non-power-of-two arises (non-square MT where `(dA+dB)/bpe` isn't 2^k), confirm the read path can emit a multiply instead of `lshl`; the read still re-applies ×bpe + its own pad, so the oracle hands it pre-bpe elems, not bytes.
- **Read `+1 / maxLDSConstOffset` (Plus 64K)** — confirm the read addressing for A1 in the next segment stays within the ds 16-bit immediate limit / uses the +1 register scheme correctly.
- **footprint reuse** — the oracle must compute `pad()` with the *same* `ldsBlockSizePerPad{tc}`/`LdsPad{tc}` the codegen uses, so footprint == baseline woffset stride exactly.
- Shared codegen: mitigated by flag-off = byte-identical + env off-switch; tight branch is same-size + monotone → lowest risk.

## 9. Later phases (not in this spec's implementation)
- **Phase 2 — aligned branch** (small MacroTile): `A1 = ceil_to_SEG(LdsOffsetA + footprintA + footprintB)`; update LDS total, `blockOffset`, double-buffer swap constants, budget (≤5×SEG) and occupancy consideration.
- **Phase 3 — tile-major (tlu)** kernels.
