# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 LDS segment-conflict interleave oracle.

Pure function of `state` that decides whether a wave-separated TDM kernel should
split its A/B halves across LDS segments (so the two MFMA read ports hit different
segments), and returns the offsets the emit sites consume.
"""

import os

# gfx1250 LDS segment size (5 x 64 KiB segments).
SEG = 65536

def _off_switch_disabled():
    return os.environ.get("TENSILE_LDS_SEGMENT_INTERLEAVE", "1").lower() in ("0", "false", "off")

def _bpe(state):
    # DataType is a DataType object, not a string; numBytes() is 2.0 for bf16.
    return int(state["ProblemType"]["DataType"].numBytes())

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
    # Each read port must cover one contiguous group per tensor; check A and B, since a
    # fine VW (e.g. VWB=1 from an odd WaveTile) reads stripes spanning both wave-halves.
    numComp = state["NumWaves"] // 2
    mi_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    coarseA = mi_threads * state["VectorWidthA"] >= state["MacroTile0"] // numComp
    coarseB = mi_threads * state["VectorWidthB"] >= state["MacroTile1"] // numComp
    return coarseA and coarseB

def _no(reason):
    return {"applicable": False, "aligned": False, "offsets": None,
            "blockSpan": 0, "reason": reason, "segmentMap": ""}

def _ceil_seg(x):
    # round a byte offset up to the next segment boundary
    return ((x + SEG - 1) // SEG) * SEG

def aligned_budget_ok(blockSpan, numLdsBlk, naturalOffsetBlk, maxLDS):
    """Return (ok, per-buffer block) for the aligned branch: the block is the next power
    of two >= max(naturalOffsetBlk, blockSpan), valid only if double-buffering it fits MaxLDS."""
    if numLdsBlk != 2:
        return (False, None)
    offsetBlk = max(naturalOffsetBlk, blockSpan)
    if offsetBlk <= 0:
        return (False, None)
    roundup = 1 << (offsetBlk - 1).bit_length()   # next power of two
    if roundup * 2 > maxLDS:
        return (False, None)                      # total = roundup + blockSpan <= roundup*2
    return (True, roundup)

def evaluate(state):
    pt = state["ProblemType"]
    if _off_switch_disabled():                                  return _no("off-switch")
    if not (state.get("enableTDMA") and state.get("enableTDMB") and state["NumWaves"] > 1):
        return _no("not wave-separated TDM")
    if not state.get("UnrollMajorLDSA") or not state.get("UnrollMajorLDSB"):
        return _no("not unrollMajor (tile-major deferred)")  # the LDS-layout flag, matches isLDSTrEnabled
    if state["NumWaves"] // 2 != 2:                             return _no("numComp!=2")
    if state.get("TDMSplit") or pt.get("MXBlockA") or pt.get("MXBlockB") or pt.get("Sparse"):
        return _no("split/mxs/sparse")
    _dt = pt["DataType"]
    if not (_dt.isBFloat16() or _dt.isHalf() or _dt.is8bitFloat()):
        return _no("v1: bf16/fp16/fp8 only")  # offset math is bpe-generic; fp8 (LDSTr off) shares the read path
    if not _coarse_vw(state):                                   return _no("fine VW")

    fA, fB = _footprint(state, "A"), _footprint(state, "B")
    dA, dB = _data_bytes(state, "A"), _data_bytes(state, "B")
    base = state["LdsOffsetA"]
    bpe = _bpe(state)

    if (base % SEG) + fA + fB < SEG:
        # Small MacroTile: everything fits one segment, so push A1 to the next segment
        # boundary. This grows LDS; Solution.py inflates offsetBlk, fixes the accumulator,
        # and does the final budget check. v1 = simple double-buffer only.
        if state.get("PrefetchGlobalRead") != 2:        return _no("small MT: PGR!=2 (aligned v1 double-buffer only)")
        if state.get("1LDSBuffer") == 1:                return _no("small MT: 1LDSBuffer (aligned v1 double-buffer only)")
        if state.get("DtlPlusLdsBuf"):                  return _no("small MT: DtlPlusLdsBuf (aligned v1 unsupported)")
        if state.get("UseSubtileImpl"):                 return _no("small MT: subtile (aligned v1 unsupported)")
        pre = _ceil_seg(base + fA + fB) - base          # pre-pad half-stride (== SEG for base<SEG)
        # Write and read each re-apply their own block pad, so A1 and B1 shift by different
        # amounts when padA != padB; postA/postB are those post-pad shifts.
        postA = pre + _pad(pre, state["LdsBlockSizePerPadA"], state["LdsPadA"], bpe)
        postB = pre + _pad(pre, state["LdsBlockSizePerPadB"], state["LdsPadB"], bpe)
        # Shift B's base by the pad gap so B1 lands right after A1 (equal pad -> gap 0).
        ldsBaseB = base + fA + max(0, postA - postB)
        offsets = {
            "ldsBaseB":         ldsBaseB,               # B0 after A0 in seg0; B1 after A1 in seg1
            "writeStrideBytes": pre,                    # pre-pad (shared); KWA re-adds each tc's pad
            "readWaveStride":   pre // bpe,             # pre-bpe elems; read re-applies *bpe + pad
        }
        # blockSpan = real post-pad per-buffer span (max of A1-end and B1-end).
        blockSpan = base + fA + max(postA, postB) + fB
        return {"applicable": True, "aligned": True, "offsets": offsets,
                "blockSpan": blockSpan, "reason": "aligned",
                "segmentMap": "ALIGNED seg%d={A0,B0} seg%d={A1,B1}"
                              % (base // SEG, (base + postA) // SEG)}

    offsets = {
        "ldsBaseB":         base + fA,          # post-pad flat
        "writeStrideBytes": dA + dB,            # pre-pad
        "readWaveStride":   (dA + dB) // bpe,   # pre-bpe
    }
    a0 = base // SEG
    a1 = (base + fA + fB) // SEG
    seg_map = "CLEAN seg%d={A0,B0} seg%d={A1,B1}" % (a0, a1) if a0 != a1 else "PARTIAL"
    return {"applicable": True, "aligned": False, "offsets": offsets,
            "blockSpan": 0, "reason": "tight", "segmentMap": seg_map}
