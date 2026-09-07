# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile-time geometry for the TDM iterate-mode free-dimension edge shift.

Iterate mode walks a tensor in `tile_dim1`-row steps from one descriptor base.
`tensor_dim1` bounds the walk from that base and is not re-based per step, so a
component left with a row count that is not a whole number of steps would read
past the tensor. The edge shift pulls that component's base back by `delta` rows
so its walk ends exactly on the boundary, and the accumulators of the one MI
wave that covers those rows are moved back afterwards.

The move is confined to a single wave, which is what makes `ds_bpermute` enough,
so the shifted component must coincide with exactly one wave's span along the
coalesced free dimension.
"""

import math


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def coalLayout(kernel, isA: bool) -> dict:
    """Accumulator layout along the coalesced free dimension.

    Mirrors the quantities ShiftVectorComponentsMFMA derives, so all four
    (SourceSwap x A/B) combinations are covered by one code path.
    """
    numThreadInWave = kernel["WavefrontSize"]
    matrixInstM = kernel["MatrixInstM"]
    matrixInstN = kernel["MatrixInstN"]
    matrixInstBM = kernel["MatrixInstBM"]
    matrixInstBN = kernel["MatrixInstBN"]

    matrixInstCoal = matrixInstM if isA else matrixInstN
    matrixInstPrep = matrixInstN if isA else matrixInstM
    matrixInstBCoal = matrixInstBM if isA else matrixInstBN
    matrixInstBPrep = matrixInstBN if isA else matrixInstBM
    miWaveTileCoal = kernel["MIWaveTile"][0] if isA else kernel["MIWaveTile"][1]
    miWaveTilePrep = kernel["MIWaveTile"][1] if isA else kernel["MIWaveTile"][0]
    miWaveGroupCoal = kernel["MIWaveGroup"][0] if isA else kernel["MIWaveGroup"][1]
    vectorWidth = kernel["VectorWidthA"] if isA else kernel["VectorWidthB"]

    conThInProcDim = bool(kernel["SourceSwap"]) ^ (not isA)

    threadInterval = 1 if conThInProcDim else matrixInstPrep
    numThreadInCoal = matrixInstCoal if conThInProcDim else (numThreadInWave // matrixInstPrep)
    numContOutCoal = (
        vectorWidth if conThInProcDim else kernel["MIOutputVectorWidth"] * vectorWidth
    )

    outBlocksInMI = (
        1
        if conThInProcDim
        else (vectorWidth * matrixInstCoal * matrixInstPrep)
        // numThreadInWave
        // numContOutCoal
    )
    miOuterTTCoal = miWaveTileCoal // vectorWidth

    # A thread's coalesced-dimension registers form `miOuterTTCoal` runs of
    # `numContOutCoal * OutBlocksInMI * matrixInstBCoal`; consecutive runs sit
    # `WGShapeCoal = MIBShapeCoal * miWaveGroupCoal` apart in coordinate space.
    numRegInMIBCoal = numContOutCoal * outBlocksInMI * matrixInstBCoal

    # Same quantity as ShiftVectorComponentsMFMAAllThread's
    # `MIBShapeCoal // numThreadInCoal`, spelled the way that emitter spells it.
    subMBShapeCoal = (
        (matrixInstCoal * vectorWidth)
        if conThInProcDim
        else (numThreadInCoal * numContOutCoal)
    )
    miBShapeCoal = subMBShapeCoal * outBlocksInMI * matrixInstBCoal
    assert numRegInMIBCoal == miBShapeCoal // numThreadInCoal, (
        "numRegInMIBCoal=%u disagrees with MIBShapeCoal(%u) // numThreadInCoal(%u)"
        % (numRegInMIBCoal, miBShapeCoal, numThreadInCoal)
    )

    numOutputsPrep = (matrixInstCoal * matrixInstPrep // numThreadInWave) if conThInProcDim else 1
    numOutputsPrep = numOutputsPrep * matrixInstBPrep * miWaveTilePrep

    # Same strides ShiftVectorComponentsMFMA uses, so both accumulator layouts
    # are addressed by one formula.
    regStrideCoal = 1 if isA else numOutputsPrep
    regStridePrep = (
        miOuterTTCoal * matrixInstBCoal * outBlocksInMI * numContOutCoal if isA else 1
    )

    return {
        "conThInProcDim": conThInProcDim,
        "threadInterval": threadInterval,
        "numThreadInCoal": numThreadInCoal,
        "numContOutCoal": numContOutCoal,
        "numOutputsPrep": numOutputsPrep,
        "regStrideCoal": regStrideCoal,
        "regStridePrep": regStridePrep,
        # Register runs along the coalesced dimension: `miOuterTTCoal` of them,
        # `numRegInMIBCoal` registers apart. The un-shift emits one block per run
        # and branches on which run holds the boundary component.
        "miOuterTTCoal": miOuterTTCoal,
        "numRegInMIBCoal": numRegInMIBCoal,
        "miWaveGroupCoal": miWaveGroupCoal,
        # Block counts the un-shift emitter does not iterate, so the oracle
        # requires a single block of each.
        "OutBlocksInMI": outBlocksInMI,
        "matrixInstBCoal": matrixInstBCoal,
    }


def geometry(state: dict, tc: str) -> dict:
    """Compile-time quantities the edge shift depends on, for tensor `tc`."""
    ti = 0 if tc == "A" else 1
    numComp = state["NumWaves"] // 2
    sparse = state["ProblemType"].get("Sparse", 0)
    dim1Divisor = 2 if (state.get("TDMSplit") and not sparse) else 1
    mt = state["MacroTile%u" % ti]
    rowsPerWave = mt // numComp // dim1Divisor if numComp else 0

    # Three element sizes meet here and must agree, or the row stride this
    # geometry computes would not be the one the descriptor walks:
    #   * `ProblemType["DataType%s"]` -- the in-memory global element size, used
    #     below for `bytesPerRow` and hence `tileDim1`;
    #   * `ProblemType["MacDataType%s"]` -- the MAC-instruction element size,
    #     which Solution.py uses to decide iterate mode;
    #   * `tP["bpeGR"]` -- the global-read element size the emitter scales the
    #     pull-back by in KernelWriterAssembly.
    # For tc in {A, B} `tP["bpeGR"]` is exactly
    # `ProblemType["DataType%s"].numBytes()` (KernelWriter.getTensorParameters
    # sets it from `tpBpe(kernel, "DataType%s", tc)`, whose A/B branch returns
    # that value), so the emitter's scale and this geometry's row stride are the
    # same quantity. MacDataType may differ (ConvertAfterDS widens it), which is
    # why it only feeds the mode choice and never a byte count here.
    bpe = state["ProblemType"]["DataType%s" % tc].numBytes()
    bytesPerRow = int(round(state["DepthU"] * bpe))
    lbspp = state["LdsBlockSizePerPad%s" % tc]
    tileDim1 = lbspp // bytesPerRow if bytesPerRow and lbspp % bytesPerRow == 0 else 0

    miBShape = (
        state["MatrixInstM"] * state["MatrixInstBM"]
        if ti == 0
        else state["MatrixInstN"] * state["MatrixInstBN"]
    )
    waveBlockSpan = state["VectorWidth%s" % tc] * miBShape

    return {
        "numComp": numComp,
        "rowsPerWave": rowsPerWave,
        "tileDim1": tileDim1,
        "waveBlockSpan": waveBlockSpan,
        "bytesPerRow": bytesPerRow,
    }


def evaluate(state: dict, tc: str, miArchVgpr=None) -> dict:
    """Decide whether the edge shift can replace the AssertFree multiple.

    The shift has two halves and both must be expressible, so the guards below
    come in three groups: is this tensor even in scope, can the load side pull
    the base back, and can the store side move the accumulators back.

    `miArchVgpr` overrides `state["MIArchVgpr"]`, which WMMA forces on only
    later in assignDerivedParameters; callers that run before that pass the
    effective value.
    """
    g = geometry(state, tc)

    def no(reason):
        return {**g, "applicable": False, "reason": reason}

    # ---- 1. in scope? -----------------------------------------------------
    if not state.get("_TDMIterateMode%s" % tc, False):
        return no("%s is not in iterate mode" % tc)
    if state.get("UseSubtileImpl"):
        return no("subtile builds its descriptors elsewhere")
    if not (state.get("enableTDMA") and state.get("enableTDMB") and state["NumWaves"] > 1):
        return no("edge shift needs the wave-separated TDM path")

    # ---- 2. load side: can one component's base be pulled back? -----------
    # The pull-back moves exactly one wave component. Anything that makes a
    # component span more than one contiguous row range, or that gives a tensor
    # a second descriptor the pull-back cannot reach, breaks that.
    if state["ProblemType"].get("MXBlock%s" % tc):
        return no("MX scale has its own descriptor the pull-back cannot reach")
    if state.get("TDMSplit"):
        return no("TDMSplit gives each component two disjoint row ranges")

    # delta and cOwn are emitted with masks and shifts, not divides, so the
    # quantities they are derived from have to be powers of two.
    bpe = state["ProblemType"]["DataType%s" % tc].numBytes()
    if not (bpe == int(bpe) and _is_pow2(int(bpe))):
        return no("bytesPerElement=%s is not an integer power of 2" % bpe)
    if g["tileDim1"] <= 1:
        return no("tile_dim1 <= 1 leaves nothing to shift")
    if not _is_pow2(g["tileDim1"]):
        return no("tile_dim1=%u is not a power of 2" % g["tileDim1"])
    if not _is_pow2(g["rowsPerWave"]):
        return no("rowsPerWave=%u is not a power of 2" % g["rowsPerWave"])
    if g["rowsPerWave"] % g["tileDim1"] != 0:
        return no("rowsPerWave=%u is not a multiple of tile_dim1=%u"
                  % (g["rowsPerWave"], g["tileDim1"]))

    # ---- 3. store side: can the accumulators be moved back? ---------------
    # The move uses ds_bpermute, which only reaches lanes inside one wave. So
    # the displaced region must be exactly one MI wave, and the emitter must be
    # able to address it with a single (coal, prep) loop.
    if g["rowsPerWave"] != g["waveBlockSpan"]:
        return no("rowsPerWave=%u != MI wave block span=%u, so the move would "
                  "cross waves" % (g["rowsPerWave"], g["waveBlockSpan"]))
    if not (state.get("MIArchVgpr", False) if miArchVgpr is None else miArchVgpr):
        return no("MIArchVgpr is off: ds_bpermute cannot read an AGPR")
    if state["MatrixInstM"] == 4 or state["MatrixInstN"] == 4:
        return no("MatrixInstM/N == 4 needs a remap coalLayout does not do")

    # The emitter walks one block per (coal, prep) pair; more than one block in
    # any of these would leave part of the accumulators unmoved.
    # miOuterTTCoal is MIWaveTile // VectorWidth. VectorWidthA/B are fork
    # parameters and can be set above MIWaveTile, which would truncate that
    # division and describe a register layout the accumulators do not have.
    isA = tc == "A"
    miWaveTileCoal = state["MIWaveTile"][0] if isA else state["MIWaveTile"][1]
    vectorWidthCoal = state["VectorWidth%s" % tc]
    if vectorWidthCoal <= 0 or miWaveTileCoal % vectorWidthCoal != 0:
        return no("MIWaveTileCoal=%u is not a multiple of VectorWidth%s=%u, so "
                  "miOuterTTCoal would not be a whole number of register runs"
                  % (miWaveTileCoal, tc, vectorWidthCoal))

    lay = coalLayout(state, isA)
    for key in ("OutBlocksInMI", "matrixInstBCoal"):
        if lay[key] != 1:
            return no("%s=%u: the emitter covers a single block" % (key, lay[key]))

    # With more than one register run per thread the boundary component is named
    # by a (tt, waveG0) pair rather than by the wave alone:
    #   tt = cOwn // miWaveGroupCoal, waveG0 = cOwn % miWaveGroupCoal.
    # The emitter forms that split with a shift and a mask, and the split only
    # names every component once when the two counts multiply out to numComp.
    if lay["miOuterTTCoal"] != 1:
        miWaveGroupCoal = lay["miWaveGroupCoal"]
        if not _is_pow2(miWaveGroupCoal):
            return no("miWaveGroupCoal=%u is not a power of 2, so cOwn cannot be "
                      "split with a shift and a mask" % miWaveGroupCoal)
        if lay["miOuterTTCoal"] * miWaveGroupCoal != g["numComp"]:
            return no("miOuterTTCoal=%u * miWaveGroupCoal=%u != numComp=%u, so the "
                      "(tt, wave) split does not name the components one-to-one"
                      % (lay["miOuterTTCoal"], miWaveGroupCoal, g["numComp"]))

    # A pass shifts by s within a lane and takes the top s from the neighbour.
    # s == numContOutCoal is a pure one-lane rotation and is fine; beyond that
    # the neighbour is no longer s positions away.
    maxStep = max(shift_steps(g["tileDim1"]))
    if maxStep > lay["numContOutCoal"]:
        return no("largest shift step=%u exceeds numContOutCoal=%u"
                  % (maxStep, lay["numContOutCoal"]))

    return {**g, "applicable": True, "reason": ""}


def shift_steps(tileDim1: int) -> list[int]:
    """Descending powers of two whose sum can express any delta in [1, tileDim1)."""
    return [1 << i for i in reversed(range((tileDim1 - 1).bit_length()))]


def apply_policy(state: dict, reject, miArchVgpr=None) -> bool:
    """Per iterate-mode tensor, choose the edge shift or the AssertFree multiple.

    The walk steps whole `tile_dim1` rows, so a MacroTile that is not a whole
    number of steps has no valid per-component split at all; that is rejected
    regardless of which mechanism handles the free-size remainder.

    `miArchVgpr` is forwarded to `evaluate`.

    Returns False when the solution was rejected.
    """
    for tc, freeIdx in (("A", 0), ("B", 1)):
        state["_TDMIterEdgeShift%s" % tc] = False
        if not state.get("_TDMIterateMode%s" % tc, False) or state.get("UseSubtileImpl"):
            continue
        r = evaluate(state, tc, miArchVgpr)
        if r["tileDim1"] <= 1:
            continue
        mt = state["MacroTile%u" % freeIdx]
        if mt % r["tileDim1"] != 0:
            reject(
                "TDM iterate %s: MacroTile%u(%u) is not a multiple of tile_dim1(%u)"
                % (tc, freeIdx, mt, r["tileDim1"])
            )
            return False
        if r["applicable"]:
            state["_TDMIterEdgeShift%s" % tc] = True
            continue
        key = "AssertFree%uElementMultiple" % freeIdx
        state[key] = int(math.lcm(state[key], r["tileDim1"]))
    return True
