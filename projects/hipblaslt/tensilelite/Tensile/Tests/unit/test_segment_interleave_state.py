# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
