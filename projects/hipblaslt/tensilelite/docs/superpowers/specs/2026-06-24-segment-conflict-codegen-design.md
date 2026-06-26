# Design: LDS segment-conflict interleave in codegen (gfx1250 TDM)

**Date:** 2026-06-24
**Status:** Approved (design); pending spec review → implementation plan
**Component:** TensileLite kernel generator (`projects/hipblaslt/tensilelite`)

## 1. Problem

On gfx1250, LDS is 5×64KiB **segments**. The 4 SIMDs form **2 pairs**, each served by one **port** (256 B/cycle; 512 B/cycle peak only when the two ports hit *different* segments in a cycle). Two ports hitting the same segment in the same cycle = a **segment conflict** (throughput halves).

For the gfx1250 TDM (`tensor_load_to_lds`) path, the MFMA read pairs are:
- `port0 = {W0, W2}` (even SIMDs, M-group 0)
- `port1 = {W1, W3}` (odd SIMDs, M-group 1)

Baseline LDS places both A-halves in one segment and both B-halves in the next:
```
baseline:  seg0 = {A0, A1}   seg1 = {B0, B1}
A-read:  port0 reads A0(seg0), port1 reads A1(seg0)  → both seg0 → CONFLICT
```

A verified hand-edit (compiled + validation PASSED on gfx1250) fixes the **VW8** case by interleaving A and B across segments so each port's A-data lands in a distinct segment:
```
interleave: seg0 = {A0, B0}   seg1 = {A1, B1}
A-read:  port0 reads A0(seg0), port1 reads A1(seg1)  → different → no conflict
```
Concretely the hand-edit changed four numbers: write wave-half stride `32768→65536`, write+read B-base `66048→33024`, read wave-half stride (shift `14→15`, i.e. `16384→32768`). This design ports that fix into codegen so it applies automatically to every qualifying kernel.

### Out of scope
- **VW4 / fine-interleave** (each wave reads interleaved 64-row stripes spanning *both* M-groups): the conflict there needs **M-split writes**, which the TDM path does not provide (TDMSplit splits **K**, not M). Not addressed here.
- **numComp > 2**, MXS, Sparse, TDMSplit.
- **Bank conflict** (intra-segment, 32×4B): orthogonal — controlled by `LdsPad`/VW/stride, untouched by this segment-level reorder.

## 2. Decisions (agreed)

| Decision | Choice |
|---|---|
| Activation | **Automatic** when the oracle reports feasible |
| Scope | **2-way interleave only** (VW8-style: `numComp==2` AND coarse VW) |
| Decision logic | **Standalone padding-accurate oracle module**, called at solution time; result stored in `state` |
| LDS size | **Unchanged** — interleave is a pure reorder within the already-allocated span |
| Safety net | **Env-var off-switch** (default ON) + **per-kernel APPLIED/SKIP log** |
| Verification | Oracle unit tests + end-to-end regen→build→run + inert-when-off regression |

## 3. Architecture

One decision, three consumers:
```
solution time (Solution.py setLdsOffsets):
    res = segment_interleave.evaluate(state)
    state["LDSSegInterleave"]        = res.applicable
    state["LDSSegInterleaveOffsets"] = res.offsets        # {strideA, strideB, ldsBaseB, readStrideShift}
    log(res)                                              # APPLIED map | SKIP reason

kernel emit time — each site branches on the flag:
    1) KWA write woffset       → stride ×2 for wId1
    2) KWA write/read B-base    → stored chunk-aligned base
    3) LraTileAssignment read   → strideWave ×2
```
When the flag is False (off-switch / non-qualifying / infeasible) every emit site is byte-identical to today. Because all placement numbers come from the *same* stored dict, write and read cannot diverge (the failure mode of the first hand-edit attempt).

## 4. The oracle module (`Tensile/.../segment_interleave.py`)

Pure function of `state`; no codegen side effects; unit-testable.

```python
SEG = 65536

def evaluate(state) -> Result:
    # structural pre-checks
    if off_switch_disabled():                       return no("off-switch")
    if not is_gfx1250_tdm(state):                   return no("not gfx1250 TDM")
    if state["NumWaves"] // 2 != 2:                 return no("numComp!=2")
    if state["TDMSplit"] or is_mxs(state) or sparse(state): return no("split/mxs/sparse")
    if not coarse_vw(state):                         return no("fine VW")   # MI_M_threads*VW >= mt//numComp

    # padding-accurate, per-buffer simulation
    chunks = simulate_layout(state)   # (tc,wId,buf) -> [start,end), pad accumulated on ABSOLUTE offset,
                                      # base = buf*blockOffset over ALL double-buffer/PGR buffers
    for buf in buffers(state):
        A = a_chunks(chunks, buf)
        if seg(A[0]) == seg(A[1]):                  return no(f"A0/A1 same segment buf{buf}")
        if any(straddle_spill(c) > EPS for c in A): return no(f"A straddle buf{buf}")
    if total_bytes(chunks) > 5*SEG:                 return no("LDS budget")

    return yes(build_offsets(chunks))
```

Design rules:
- **A-strict / B-tolerant:** A-chunks (which determine the port conflict) must be in distinct segments and not straddle beyond `EPS` (~one pad block). B-chunks may spill — both ports read all B regardless, so B placement does not affect the conflict.
- **Padding accumulation:** `off += pad(off)` uses the *absolute* offset, so progressive drift is modeled exactly (a flagged risk).
- **Per-buffer:** every double-buffer / PGR buffer must pass, not just buffer 0 — catches a non-64KiB-aligned `blockOffset` drifting later buffers (a flagged risk).
- **Budget:** padded total across buffers ≤ 5×64KiB.
- `build_offsets` returns the four numbers matching the verified hand-edit, derived from the simulated layout (so it generalizes to any qualifying VW8-style size, not hard-coded constants).

## 5. Emit-site changes (3 sites, flag-gated 2-liners)

1. **Write woffset** — `KernelWriterAssembly.py:18791` (both TDM-init blocks):
   `dataBytes *= 2` when flag set → wId1 wave lands a full segment away.
2. **Write/read B base** — write `ldsConstOffset` (KWA:18800) and read `+= LdsOffsetB` (KWA:5899) both use `offsets["ldsBaseB"]` (one padded chunk) instead of `kernel["LdsOffsetB"]` when flag set.
3. **Read strideWave** — `LraTileAssignment.py` transposed-MFMA site emitting `W0Stride(...)`: `strideWave *= 2` when flag set.

`kernel["LdsOffsetA/B"]` and the LDS-size accumulator are **not** modified, so allocation stays correct (interleave reorders within the same span; max offset unchanged → no OOB).

## 6. Safety net

**Off-switch** (env var, default ON), checked first in `evaluate()`:
```
TENSILE_LDS_SEGMENT_INTERLEAVE in {0,false,off}  → evaluate() returns applicable=False for all kernels
```
Total, code-free revert to pre-change codegen.

**Per-kernel log** (one line at solution time, via Tensile logger):
```
[LDSSegInterleave] <kernel>: APPLIED  seg0={A0@0,B0@33024} seg1={A1@66048,B1@99072} (chunk=33024,buffers=2)
[LDSSegInterleave] <kernel>: SKIP     reason="fine VW"
```

## 7. Verification

**Unit (no GPU, `Tensile/Tests/unit`):** synthetic `state` → assert `evaluate()`:
- VW8 MT256×256×128 bf16 → APPLIED, offsets `{stride=65536, ldsBaseB=33024, readShift=15}` (== hand-edit).
- VW4 → SKIP "fine VW"; numComp≠2 → SKIP; TDMSplit/MXS/Sparse → SKIP; off-switch → SKIP all.
- Pad-accumulation + non-64KiB `blockOffset` config whose buffer-1 A-chunk would straddle → SKIP (guards the two flagged risks).

**End-to-end (GPU loop):**
1. Regenerate the BBS kernel through the pipeline; inspect `.s` for `woffset 65536`, B base `0x8100`, read shift `15` (byte-equivalent to hand-edit).
2. `make co TENSILE_OUT=my-custom-build TARGET_ARCH=gfx1250 ARCH=gfx1250 WAVE=32` → `run.sh` → validation **PASSED**.
3. Regression: a non-qualifying kernel (VW4) generated `.s` diff vs pre-change = empty (flag-off path inert).
4. (Optional) perf compare baseline vs interleave.

**Open item for the plan:** exact single-kernel regeneration command (heavier than `make co`).

## 8. Risks

- Shared codegen: mitigated by flag-off = byte-identical + env off-switch.
- Padding accumulation / double-buffer drift: handled by the per-buffer, absolute-offset simulation; unit tests assert rejection of unsafe configs.
- `strideWave` has 5 emit sites; only the gfx1250 transposed-MFMA TDM one changes — confirm by matching the generated comment text during implementation.
