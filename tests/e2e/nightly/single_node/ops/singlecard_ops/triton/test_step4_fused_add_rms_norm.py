#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""NPU test: Step4 fused add-RMSNorm-cast Triton kernel vs eager reference.

Uses the actual Step-4-preview-2608 checkpoint configuration:
hidden_size=4096, bf16 activations (torch_dtype), fp32 norm weight
(norm_dtype), fp32 residual (fp32_residual_connection=true),
zero_centered=True (config default), eps=1e-5 (rms_norm_eps).
"""

import pytest
import torch

from vllm_ascend.models.step4.fused_add_rms_norm import (
    _fused_add_rms_norm_cast_eager,
    fused_add_rms_norm_cast,
)

DEVICE = "npu"

# Actual Step-4-preview-2608 config.json values.
STEP4_HIDDEN_SIZE = 4096
STEP4_EPS = 1e-5
STEP4_DTYPE = torch.bfloat16  # torch_dtype (activations / param-op dtype)
STEP4_WEIGHT_DTYPE = torch.float32  # norm_dtype
STEP4_ZERO_CENTERED = True  # config default when absent
STEP4_FP32_RESIDUAL = True  # fp32_residual_connection

ULP = {
    torch.float16: 2**-10,
    torch.bfloat16: 2**-8,
    torch.float32: 2**-23,
}


def _max_ulp_diff(actual, expected):
    diff = (actual.float() - expected.float()).abs().max().item()
    magnitude = max(expected.float().abs().max().item(), 1.0)
    return diff / (ULP[actual.dtype] * magnitude)


@pytest.mark.parametrize("num_tokens", [1, 7, 65, 128])
def test_fused_add_rms_norm_cast_matches_eager(num_tokens):
    torch.manual_seed(17)
    hidden_size = STEP4_HIDDEN_SIZE
    x = (torch.randn(num_tokens, hidden_size) * 0.1).to(STEP4_DTYPE).to(DEVICE)
    residual = (
        (torch.randn(num_tokens, hidden_size) * 0.1).to(torch.float32).to(DEVICE)
    )
    weight = (torch.randn(hidden_size) * 0.1).to(STEP4_WEIGHT_DTYPE).to(DEVICE)

    out, residual_out = fused_add_rms_norm_cast(
        x,
        residual,
        weight,
        STEP4_EPS,
        zero_centered=STEP4_ZERO_CENTERED,
        fp32_residual=STEP4_FP32_RESIDUAL,
    )
    expected_out, expected_residual = _fused_add_rms_norm_cast_eager(
        x, residual, weight, STEP4_EPS, STEP4_ZERO_CENTERED, STEP4_FP32_RESIDUAL
    )

    assert out.dtype == torch.bfloat16
    assert residual_out.dtype == torch.float32
    # The fp32->low-precision rounding matches torch on device, so outputs
    # must agree to within one ulp of the storage dtype.
    assert _max_ulp_diff(residual_out, expected_residual) <= 1.0
    assert _max_ulp_diff(out, expected_out) <= 1.0
