# Design: LDS segment-conflict interleave in codegen (gfx1250 TDM)

**Date:** 2026-06-24
**Status:** Approved (design); pending spec review → implementation plan
**Component:** TensileLite kernel generator (`projects/hipblaslt/tensilelite`)

## 1. Problem

On gfx1250, LDS is 5×64KiB **segments** (`SEG = 65536`). The 4 SIMDs form **2 pairs**, each served by one **port** (256 B/cycle; 512 B/cycle peak only when the two ports hit *different* segments in a cycle). Two ports hitting the same segment in the same cycle = a **segment conflict** (throughput halves).

For the gfx1250 TDM (`tensor_load_to_lds`) path the MFMA read ports are:
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
Inserting B between A0 and A1 increases the port-to-port distance for the A-read from `< SEG` (baseline: A0/A1 same segment, ~100% of concurrent ds_load pairs collide) to `≥ SEG` (A1 pushed ≥1 segment away; at worst a small boundary-straddle overlap). So the A-read conflict count can only **decrease or stay equal**, never increase. B was already maximally collided in baseline (all in one segment); spreading it cannot make it worse. The reorder is bank-neutral (shifts are multiples of 128 B) and, in the tight case, keeps total LDS unchanged. ⇒ We need only ensure **not worse**, not prove **zero conflict**.

### Out of scope
- **VW4 / fine-interleave** (each wave reads interleaved 64-row stripes spanning *both* M-groups): needs **M-split writes**, which the TDM path does not provide (TDMSplit splits **K**, not M). Not addressed here.
- **numComp > 2**, MXS, Sparse, TDMSplit.
- **Bank conflict** (intra-segment, 32×4 B): orthogonal — controlled by `LdsPad`/VW/stride, untouched by this segment-level reorder.

## 2. Decisions (agreed)

| Decision | Choice |
|---|---|
| Activation | **Automatic** when the oracle reports feasible |
| Scope | **2-way interleave** (numComp==2, coarse VW): **tight** for large MacroTile, **aligned** for small MacroTile |
| Decision logic | **Standalone oracle module**, called at solution time; result (flag + offsets + segment map) stored in `state` |
| Safety net | **Env-var off-switch** (default ON) + **per-kernel log** (CLEAN / PARTIAL / SKIP) |
| Verification | Oracle unit tests + end-to-end regen→build→run + inert-when-off regression |

## 3. Architecture

One decision, three consumers (single source of truth → write and read cannot diverge, the bug that broke the first hand-edit):
```
solution time (Solution.py setLdsOffsets):
    res = segment_interleave.evaluate(state)
    state["LDSSegInterleave"]        = res.applicable
    state["LDSSegInterleaveOffsets"] = res.offsets    # {strideA, strideB, ldsBaseB, readWaveStride}
    log(res)                                           # CLEAN/PARTIAL map | SKIP reason

kernel emit time — each site branches on the flag and uses the stored offsets:
    1) KWA write woffset stride       2) KWA write/read B-base       3) LraTileAssignment read wave-stride
```
Flag False (off-switch / non-qualifying / infeasible) ⇒ every emit site byte-identical to today.

## 4. The oracle (`Tensile/.../segment_interleave.py`)

Pure function of `state`; no side effects; unit-testable. **Decision tree:**

```python
SEG = 65536

def evaluate(state) -> Result:
    # ---- gate: structural qualifying conditions ----
    if off_switch_disabled():                                  return skip("off-switch")
    if not is_gfx1250_tdm(state):                              return skip("not gfx1250 TDM")
    if state["NumWaves"] // 2 != 2:                            return skip("numComp!=2")
    if state["TDMSplit"] or is_mxs(state) or sparse(state):    return skip("split/mxs/sparse")
    if not coarse_vw(state):    # MI_M_threads*VW >= mt//numComp (each port = single M-group)
                                                               return skip("fine VW")

    c = padded_chunk(state)     # (mt//numComp)*du*bpe + pad

    # ---- branch by whether A0+B0 reach a full segment ----
    if 2*c >= SEG:
        # LARGE MacroTile: tight pure reorder. A1 lands in the next segment naturally.
        offsets = tight_offsets(state, c)       # strideA/B = 2*dataChunk, ldsBaseB = one chunk
        # total LDS unchanged → always fits if baseline fit; monotone better → apply unconditionally
        return apply(offsets, segment_map(state, offsets))     # map → CLEAN or PARTIAL (for log)
    else:
        # SMALL MacroTile: tight would leave A1 in seg0 (no separation).
        # Force A1 (wId1 wave) to the next SEG boundary; costs LDS (gap after B0).
        offsets = aligned_offsets(state, c)     # strideA/B = SEG, ldsBaseB = one chunk
        if double_buffer_total(state, offsets) > 5*SEG:        return skip("aligned exceeds LDS budget")
        return apply(offsets, segment_map(state, offsets))     # aligned ⇒ A0/A1 each clean, distinct
```

Design rules:
- **Two branches keyed on `2*c >= SEG`.**
  - **Tight** (large MT): A1 at `2*dataChunk` → already in the next segment. **Same total LDS** (pure reorder of the same span). Monotone better ⇒ **apply unconditionally**, no budget gate. The seg boundary falls on B0 (tolerant); for very large `c` A1 may also straddle slightly, still ≤ baseline.
  - **Aligned** (small MT, `2*c < SEG`): force A1 to the next `SEG` boundary so A0/A1 land in distinct segments (each clean, no straddle). **Grows LDS** (gap after B0) ⇒ **budget-gated**: apply only if double-buffer total ≤ 5×SEG, else baseline.
- **No EPS, no per-buffer straddle gate.** These tried to *prove zero conflict*; the monotone property means we only need *not worse* (tight) or *clean by construction* (aligned). Straddle in the tight case is benign (still ≤ baseline) and B-straddle is irrelevant (both ports read all B anyway).
- **`segment_map`** computes the padding-accurate per-buffer segment placement **only to label the log** (`CLEAN` = A fully separated in every buffer; `PARTIAL` = some A straddle/overlap but still ≤ baseline). It is **not** a gate.
- **`build_offsets`** returns the four numbers consumed by the emit sites, derived from the chosen layout (so it generalizes to any qualifying size, not hard-coded constants). For VW8 tight these equal the verified hand-edit.

## 5. Emit-site changes (3 sites, flag-gated, consume stored offsets)

1. **Write woffset stride** — `KernelWriterAssembly.py:18791` (both TDM-init blocks): use `offsets["strideA"|"strideB"]` instead of `dataBytes` when flag set.
2. **Write/read B base** — write `ldsConstOffset` (KWA:18800) and read `+= LdsOffsetB` (KWA:5899) use `offsets["ldsBaseB"]` instead of `kernel["LdsOffsetB"]` when flag set.
3. **Read wave-stride** — `LraTileAssignment.py` transposed-MFMA site (emits `W0Stride(...)`): use `offsets["readWaveStride"]` instead of `strideWave` when flag set.

`kernel["LdsOffsetA/B"]` and the LDS-size accumulator are **not** modified — interleave reorders within the allocated span (tight) or the oracle has already budget-checked the larger span (aligned). Emit sites are dumb consumers; tight vs aligned differ only in the stored numbers.

## 6. Safety net

**Off-switch** (env var, default ON), checked first in `evaluate()`:
```
TENSILE_LDS_SEGMENT_INTERLEAVE in {0,false,off}  → evaluate() returns applicable=False for all kernels
```
Total, code-free revert to pre-change codegen.

**Per-kernel log** (one line at solution time, via Tensile logger):
```
[LDSSegInterleave] <kernel>: CLEAN   tight    seg0={A0@0,B0@33024} seg1={A1@66048,B1@99072}
[LDSSegInterleave] <kernel>: CLEAN   aligned  seg0={A0@0,B0@16640} seg1={A1@65536,B1@82176}
[LDSSegInterleave] <kernel>: PARTIAL tight    A1 straddles seg1/seg2 in buf0 (still <= baseline)
[LDSSegInterleave] <kernel>: SKIP    reason="fine VW"
```

## 7. Verification

**Unit (no GPU, `Tensile/Tests/unit`):** synthetic `state` → assert `evaluate()`:
- VW8 MT256×256×128 bf16 (large, 2c≥SEG) → APPLIED tight, offsets `{strideA/B=65536, ldsBaseB=33024, readWaveStride→shift 15}` (== verified hand-edit), log CLEAN.
- Small MT (e.g. MT128×128×128 bf16, 2c<SEG) → APPLIED aligned, A1 at next SEG boundary, distinct clean segments, budget OK.
- Small MT that overflows budget → SKIP "aligned exceeds LDS budget".
- VW4 → SKIP "fine VW"; numComp≠2 → SKIP; TDMSplit/MXS/Sparse → SKIP; off-switch → SKIP all.
- A config whose tight layout makes A1 straddle (large c) → APPLIED tight, log PARTIAL (asserts we apply, not reject).

**End-to-end (GPU loop):**
1. Regenerate the BBS (VW8) kernel through the pipeline; inspect `.s` for `woffset 65536`, B base `0x8100`, read shift `15` (byte-equivalent to hand-edit).
2. `make co TENSILE_OUT=my-custom-build TARGET_ARCH=gfx1250 ARCH=gfx1250 WAVE=32` → `run.sh` → validation **PASSED**.
3. Regression: a non-qualifying kernel (VW4) generated `.s` diff vs pre-change = empty (flag-off path inert).
4. (Optional) perf compare baseline vs interleave.

**Open item for the plan:** exact single-kernel regeneration command (heavier than `make co`).

## 8. Risks

- Shared codegen: mitigated by flag-off = byte-identical + env off-switch.
- **Tight branch** is provably not-worse (monotone) and same-size → lowest risk, no budget concern.
- **Aligned branch** grows LDS → the budget gate is load-bearing; also subject to double-buffer drift, but by the monotone property even a drifted buffer stays ≤ baseline (drift affects optimality, not safety).
- `strideWave` has 5 emit sites; only the gfx1250 transposed-MFMA TDM one changes — confirm by matching the generated comment text during implementation.
- `readWaveStride` is emitted as a shift when it is a power of two; the oracle must hand the emit site a value/representation the generator can encode (shift vs multiply) — pin down in the plan.
