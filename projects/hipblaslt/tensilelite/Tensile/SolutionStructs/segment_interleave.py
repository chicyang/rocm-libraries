# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
