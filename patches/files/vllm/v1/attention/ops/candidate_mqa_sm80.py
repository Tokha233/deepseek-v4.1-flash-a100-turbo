# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's mqa_logits_triton.py; sparse selection follows SGLang #38944.
"""SM80 candidate-only index logits, preserving the dense logical-column ABI."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.mqa_logits_triton import _get_e4m3fn_bf16_lut


@triton.jit(do_not_specialize=["width"])
def _candidate_logits(
    Q, K, S, W, LUT, LENS, TABLE, CANDIDATES, OUT,
    stride_qr, stride_qh, stride_qd, stride_kb, stride_ks, stride_kd,
    stride_sb, stride_ss, stride_wr, stride_wh, stride_lr,
    stride_tr, stride_tc, stride_cr, stride_cc, stride_or,
    width, NC: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, CB: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BN + tl.arange(0, BN)
    cid = offsets // CB
    candidate = tl.load(CANDIDATES + row * stride_cr + cid * stride_cc, cid < NC, -1)
    position = candidate * CB + offsets % CB
    length = tl.load(LENS + row * stride_lr)
    valid = (cid < NC) & (candidate >= 0) & (position < length) & (position < width)
    if tl.sum(valid.to(tl.int32), 0) == 0:
        return
    physical = tl.load(TABLE + row * stride_tr + position // PAGE * stride_tc, valid, 0).to(tl.int64)
    hh = tl.arange(0, BH)
    dd = tl.arange(0, BD)
    q = tl.load(Q + row * stride_qr + hh[:, None] * stride_qh + dd[None, :] * stride_qd,
                (hh[:, None] < H) & (dd[None, :] < D), 0)
    kb = tl.load(K + physical[:, None] * stride_kb + (position % PAGE)[:, None] * stride_ks
                 + dd[None, :] * stride_kd, valid[:, None] & (dd[None, :] < D), 0)
    k = tl.load(LUT + kb.to(tl.uint32))
    scale = tl.load(S + physical * stride_sb + position % PAGE * stride_ss, valid, 0.0)
    scores = tl.dot(q, tl.trans(k)) * scale[None, :]
    weight = tl.load(W + row * stride_wr + hh * stride_wh, hh < H, 0.0)
    scores = tl.where(scores > 0, scores, 0.0) * weight[:, None]
    result = tl.sum(scores, 0)
    tl.store(OUT + row * stride_or + position, result, valid)


def candidate_mqa_logits(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor,
                         lengths: torch.Tensor, table: torch.Tensor, candidates: torch.Tensor,
                         width: int, candidate_block_size: int = 8) -> torch.Tensor:
    """Accept flattened decode rows, unique source-selected block IDs and -1 padding."""
    rows, next_n, heads, dimension = q.shape
    assert next_n == 1 and lengths.shape == (rows, 1)
    assert candidates.shape[0] == rows and candidate_block_size == 8
    assert q.dtype == torch.float8_e4m3fn and kv.dtype == torch.uint8
    assert torch.cuda.get_device_capability(q.device) == (8, 0)
    blocks, page, one, packed_dimension = kv.shape
    assert one == 1 and packed_dimension == dimension + 4
    flat = kv.view(blocks, -1)
    key_bytes = flat[:, :page * dimension].as_strided(
        (blocks, page, dimension), (flat.stride(0), dimension, 1))
    scales = flat[:, page * dimension:].view(torch.float32)
    lut = _get_e4m3fn_bf16_lut(q.device)
    query = lut.index_select(0, q.view(torch.uint8).reshape(-1).to(torch.int32)).view(rows, heads, dimension)
    output = torch.full((rows, width), -float('inf'), device=q.device, dtype=torch.float32)
    _candidate_logits[(rows, triton.cdiv(candidates.shape[1] * candidate_block_size, 64))](
        query, key_bytes, scales, weights, lut, lengths, table, candidates, output,
        *query.stride(), *key_bytes.stride(), *scales.stride(), *weights.stride(), lengths.stride(0),
        *table.stride(), *candidates.stride(), output.stride(0),
        width, candidates.shape[1], heads, dimension, page, candidate_block_size,
        max(16, triton.next_power_of_2(heads)), triton.next_power_of_2(dimension), 64,
        num_warps=4, num_stages=2,
    )
    return output
