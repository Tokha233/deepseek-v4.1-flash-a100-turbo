# SPDX-License-Identifier: Apache-2.0
"""Stable MoE token alignment using tiled integer sorting and prefix sums."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=["N"])
def _sort_count(ids, expert_map, ordered, counts, N, E: tl.constexpr,
                MAP_SIZE: tl.constexpr, TILE: tl.constexpr, BINS: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * TILE + tl.arange(0, TILE)
    expert = tl.load(ids + offsets, offsets < N, -1).to(tl.int64)
    if MAP_SIZE:
        in_range = (expert >= 0) & (expert < MAP_SIZE) & (offsets < N)
        expert = tl.load(expert_map + expert, in_range, -1).to(tl.int64)
    valid = (offsets < N) & (expert >= 0) & (expert < E)
    key = tl.where(valid, expert, E).to(tl.int32)
    histogram = tl.histogram(key, BINS)
    bins = tl.arange(0, BINS)
    tl.store(counts + tile * E + bins, histogram, bins < E)
    packed = (key.to(tl.int64) << 32) | offsets.to(tl.int64)
    tl.store(ordered + tile * TILE + tl.arange(0, TILE), tl.sort(packed, descending=False))


@triton.jit(do_not_specialize=["TILES"])
def _scan_counts(counts, offsets, totals, E: tl.constexpr, TILES, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    tiles = tl.arange(0, BLOCK)
    count = tl.load(counts + tiles * E + expert, tiles < TILES, 0)
    cumulative = tl.cumsum(count)
    tl.store(offsets + tiles * E + expert, cumulative - count, tiles < TILES)
    tl.store(totals + expert, tl.sum(count))


@triton.jit(do_not_specialize=["N", "SORTED_SIZE", "EXPERT_SIZE"])
def _bases_initialize(totals, bases, sorted_ids, expert_ids, padded_total,
                      N, E: tl.constexpr, PAD: tl.constexpr,
                      SORTED_SIZE, EXPERT_SIZE,
                      EXPERT_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        experts = tl.arange(0, EXPERT_BLOCK)
        count = tl.load(totals + experts, experts < E, 0)
        padded = tl.cdiv(count, PAD) * PAD
        cumulative = tl.cumsum(padded)
        tl.store(bases + experts, cumulative - padded, experts < E)
        tl.store(padded_total, tl.sum(padded))
    indices = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_ids + indices, N, indices < SORTED_SIZE)
    tl.store(expert_ids + indices, -1, indices < EXPERT_SIZE)


@triton.jit
def _scatter_stable(ordered, offsets, bases, sorted_ids, expert_ids,
                    E: tl.constexpr, PAD: tl.constexpr, TILE: tl.constexpr):
    tile = tl.program_id(0)
    positions = tl.arange(0, TILE)
    packed = tl.load(ordered + tile * TILE + positions)
    expert = (packed >> 32).to(tl.int32)
    original = (packed & 0xFFFFFFFF).to(tl.int32)
    previous = tl.gather(expert, tl.maximum(positions - 1, 0), axis=0)
    starts = tl.where((positions == 0) | (expert != previous), positions, 0)
    segment_start = tl.associative_scan(starts, axis=0, combine_fn=_maximum)
    valid = expert < E
    prior = tl.load(offsets + tile * E + expert, valid, 0)
    rank = prior + positions - segment_start
    base = tl.load(bases + expert, valid, 0)
    destination = base + rank
    tl.store(sorted_ids + destination, original, valid)
    tl.store(expert_ids + destination // PAD, expert, valid & (rank % PAD == 0))


def stable_moe_align(topk_ids: torch.Tensor, num_experts: int, block_size: int,
                     sorted_ids: torch.Tensor, expert_ids: torch.Tensor,
                     num_tokens_post_pad: torch.Tensor, expert_map: torch.Tensor | None) -> None:
    assert topk_ids.is_contiguous() and topk_ids.dtype in (torch.int32, torch.int64)
    assert num_experts > 0 and topk_ids.numel() < 2**31
    n, tile = topk_ids.numel(), 256
    tiles = triton.cdiv(n, tile)
    ordered = torch.empty(tiles * tile, dtype=torch.int64, device=topk_ids.device)
    counts = torch.empty((tiles, num_experts), dtype=torch.int32, device=topk_ids.device)
    offsets = torch.empty_like(counts)
    totals = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    bases = torch.empty_like(totals)
    if tiles:
        _sort_count[(tiles,)](topk_ids, expert_map, ordered, counts, n, num_experts,
                              expert_map.numel() if expert_map is not None else 0,
                              tile, triton.next_power_of_2(num_experts + 1), num_warps=4)
    _scan_counts[(num_experts,)](counts, offsets, totals, num_experts, tiles,
                                 triton.next_power_of_2(max(tiles, 1)), num_warps=4)
    _bases_initialize[(max(triton.cdiv(sorted_ids.numel(), 1024), 1),)](
        totals, bases, sorted_ids, expert_ids, num_tokens_post_pad, n, num_experts,
        block_size, sorted_ids.numel(), expert_ids.numel(),
        triton.next_power_of_2(num_experts), 1024, num_warps=4)
    if tiles:
        _scatter_stable[(tiles,)](ordered, offsets, bases, sorted_ids, expert_ids,
                                  num_experts, block_size, tile, num_warps=4)
