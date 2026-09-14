# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class MarlinMxfp8LinearKernel(Mxfp8LinearKernel):
    """MXFP8 W8A16 GEMM via Marlin (SM80+)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            is_fp8_marlin_supported,
        )

        if is_fp8_marlin_supported():
            return True, None
        return False, "Marlin FP8 not available"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_mxfp8_layer_for_marlin,
        )

        layer.ampere_bf16_min_tokens = int(os.environ.get("VLLM_AMPERE_DENSE_BF16_MIN_TOKENS", "0"))
        if layer.ampere_bf16_min_tokens and torch.cuda.get_device_capability(layer.weight.device) == (8, 0):
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

            n, k = layer.weight.shape
            layer.register_buffer("ampere_bf16_weight", dequant_mxfp8_to_bf16(
                layer.weight, layer.weight_scale[:n, :k // 32].contiguous().view(torch.uint8)), persistent=False)
            layer.register_buffer("ampere_bf16_bias", layer.bias.clone() if getattr(layer, "bias", None) is not None else None,
                                  persistent=False)
        else:
            layer.ampere_bf16_min_tokens = 0

        prepare_mxfp8_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_mxfp8_marlin_linear,
        )

        if layer.ampere_bf16_min_tokens and x.numel() // x.shape[-1] >= layer.ampere_bf16_min_tokens:
            return torch.nn.functional.linear(x, layer.ampere_bf16_weight,
                                               layer.ampere_bf16_bias if bias is not None else None)

        return apply_mxfp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
