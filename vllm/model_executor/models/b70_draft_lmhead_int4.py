# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental reduced-vocabulary INT4 draft head for B70."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LocalVocabSelection:
    global_ids: tuple[int, ...]
    local_ids: tuple[int, ...]
    padded_local_ids: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.local_ids)

    @property
    def padded_count(self) -> int:
        return len(self.padded_local_ids)


@dataclass(frozen=True)
class DraftHeadInt4State:
    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor
    group_size: int
    local_ids: torch.Tensor
    local_shard_width: int
    org_vocab_size: int
    metadata: dict[str, object]


def select_local_vocab(
    global_ids: list[int],
    org_vocab_size: int,
    org_vocab_start: int,
    org_vocab_end: int,
) -> LocalVocabSelection:
    if not global_ids or len(global_ids) != len(set(global_ids)):
        raise ValueError("draft vocabulary must be non-empty and unique")
    if min(global_ids) < 0 or max(global_ids) >= org_vocab_size:
        raise ValueError("draft vocabulary contains an out-of-range token ID")
    if not 0 <= org_vocab_start <= org_vocab_end <= org_vocab_size:
        raise ValueError("invalid original-vocabulary shard range")

    selected = tuple(
        token_id
        for token_id in global_ids
        if org_vocab_start <= token_id < org_vocab_end
    )
    local = tuple(token_id - org_vocab_start for token_id in selected)
    if local:
        padded_count = (len(local) + 7) // 8 * 8
        padded = local + (local[-1],) * (padded_count - len(local))
    else:
        padded = ()
    return LocalVocabSelection(selected, local, padded)


def quantize_lmhead_to_int4(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if weight.ndim != 2:
        raise ValueError("draft head weight must be two-dimensional")
    num_rows, hidden_size = weight.shape
    if group_size <= 0 or hidden_size % group_size or group_size % 8:
        raise ValueError("hidden size must divide into groups of eight values")
    num_groups = hidden_size // group_size
    if num_rows == 0:
        return (
            torch.empty((hidden_size // 8, 0), dtype=torch.int32, device=weight.device),
            torch.empty((num_groups, 0), dtype=weight.dtype, device=weight.device),
            torch.tensor([8], dtype=torch.int8, device=weight.device),
            group_size,
        )

    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    packed_parts = []
    scale_parts = []
    for offset in range(0, num_rows, 4096):
        chunk = weight[offset : offset + 4096].float()
        grouped = chunk.view(chunk.shape[0], num_groups, group_size)
        scales = grouped.abs().amax(dim=-1) / 7.0
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        quantized = (
            (grouped / safe_scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        )
        values = (quantized + 8).view(chunk.shape[0], num_groups, group_size // 8, 8)
        packed_parts.append(
            (values << shifts)
            .sum(dim=-1)
            .to(torch.int32)
            .reshape(chunk.shape[0], hidden_size // 8)
        )
        scale_parts.append(scales.to(weight.dtype))

    qweight = torch.cat(packed_parts).t()
    scales = torch.cat(scale_parts).t().contiguous()
    qzeros = torch.tensor([8], dtype=torch.int8, device=weight.device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    hidden_states: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*hidden_states.shape[:-1], qweight.shape[1])


def _read_global_vocab(path: Path) -> list[int]:
    try:
        return [int(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid B70 draft vocabulary {path}: {exc}") from exc


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_b70_lmhead_int4", None) is not None:
        return

    speculative_config = getattr(model.vllm_config, "speculative_config", None)
    if getattr(speculative_config, "use_local_argmax_reduction", False):
        raise RuntimeError(
            "B70 draft head INT4 requires use_local_argmax_reduction=False"
        )
    processor = model.logits_processor
    if processor.logits_as_input:
        raise RuntimeError("B70 draft head INT4 does not support logits_as_input")
    if processor.soft_cap is not None or processor.scale != 1.0:
        raise RuntimeError("B70 draft head INT4 requires unscaled, uncapped logits")
    if processor.head_dtype not in (None, torch.float16):
        raise RuntimeError("B70 draft head INT4 requires FP16 head projection")

    head = model.lm_head
    weight = head.weight.detach()
    if weight.dtype != torch.float16:
        raise RuntimeError("B70 draft head INT4 requires FP16 source weights")
    shard = head.shard_indices
    org_vocab_size = processor.org_vocab_size
    if head.org_vocab_size != org_vocab_size or head.num_added_embeddings:
        raise RuntimeError("B70 draft head INT4 does not support added vocabulary")
    if weight.shape[0] != shard.num_elements_padded:
        raise RuntimeError("unsupported draft head shard layout")

    vocab_path_value = os.environ.get("B70_DRAFT_VOCAB_PATH")
    if not vocab_path_value:
        raise RuntimeError("B70_DRAFT_VOCAB_PATH is required")
    vocab_path = Path(vocab_path_value)
    global_ids = _read_global_vocab(vocab_path)
    selection = select_local_vocab(
        global_ids,
        org_vocab_size,
        shard.org_vocab_start_index,
        shard.org_vocab_end_index,
    )
    local_ids = torch.tensor(
        selection.local_ids, dtype=torch.long, device=weight.device
    )
    padded_ids = torch.tensor(
        selection.padded_local_ids, dtype=torch.long, device=weight.device
    )
    selected_weight = weight.index_select(0, padded_ids)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(selected_weight)
    metadata = {
        "global_vocab_size": org_vocab_size,
        "global_selected_count": len(global_ids),
        "global_vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "org_vocab_start": shard.org_vocab_start_index,
        "org_vocab_end": shard.org_vocab_end_index,
        "local_shard_width": weight.shape[0],
        "local_selected_count": selection.count,
        "local_padded_count": selection.padded_count,
        "qweight_shape": tuple(qweight.shape),
        "qweight_dtype": str(qweight.dtype),
        "scales_shape": tuple(scales.shape),
        "scales_dtype": str(scales.dtype),
        "tensor_parallel_size": head.tp_size,
        "use_local_argmax_reduction": False,
    }
    model._b70_lmhead_int4 = DraftHeadInt4State(
        qweight,
        scales,
        qzeros,
        group_size,
        local_ids,
        weight.shape[0],
        org_vocab_size,
        metadata,
    )
    print(f"[B70] TP2 draft LM head INT4 active: {metadata}", flush=True)


def draft_lmhead_int4_local_logits(state, hidden_states: torch.Tensor):
    if state.local_ids.numel():
        selected_logits = int4_lmhead_logits(
            hidden_states,
            state.qweight,
            state.scales,
            state.qzeros,
            state.group_size,
        )[..., : state.local_ids.numel()]
    else:
        selected_logits = hidden_states.new_empty((*hidden_states.shape[:-1], 0))
    local_logits = hidden_states.new_full(
        (*hidden_states.shape[:-1], state.local_shard_width), -torch.inf
    )
    if state.local_ids.numel():
        local_logits.index_copy_(-1, state.local_ids, selected_logits)
    return local_logits


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor):
    state = model._b70_lmhead_int4
    local_logits = draft_lmhead_int4_local_logits(state, hidden_states)
    logits = model.logits_processor._gather_logits(local_logits)
    if logits is None:
        return None
    logits = logits[..., : state.org_vocab_size]
    return logits
