# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the experimental vocabulary-sharded draft head."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.models import b70_draft_lmhead_int4 as head


def unpack(qweight, scales, group_size):
    packed = qweight.t().to(torch.int64)
    columns = [((packed >> shift) & 15) - 8 for shift in range(0, 32, 4)]
    values = torch.stack(columns, dim=-1).reshape(packed.shape[0], -1)
    return values.float() * scales.t().float().repeat_interleave(group_size, -1)


def test_global_tokens_map_to_their_own_shard_and_padding_is_discardable():
    ids = [28, 3, 19, 7, 30]
    a = head.select_local_vocab(ids, 32, 0, 16)
    b = head.select_local_vocab(ids, 32, 16, 32)
    assert a.local_ids == (3, 7)
    assert b.local_ids == (12, 3, 14)
    assert a.padded_local_ids == (3, 7, 7, 7, 7, 7, 7, 7)
    assert b.padded_count == 8
    assert set(a.global_ids + b.global_ids) == set(ids)
    empty = head.select_local_vocab([3, 7], 32, 16, 32)
    assert empty.count == empty.padded_count == 0


@pytest.mark.parametrize("ids", [[], [2, 2], [-1], [32]])
def test_invalid_global_vocabulary_is_rejected(ids):
    with pytest.raises(ValueError):
        head.select_local_vocab(ids, 32, 0, 16)


def test_packed_signed_nibbles_and_zero_weight_groups_round_trip():
    weight = torch.arange(-7, 9).clamp(-7, 7).repeat(8).half().repeat(8, 1)
    weight[0].zero_()
    original = weight.clone()
    qweight, scales, qzeros, group_size = head.quantize_lmhead_to_int4(weight)
    assert torch.equal(unpack(qweight, scales, group_size), weight.float())
    assert torch.equal(weight, original)
    assert torch.isfinite(scales).all()
    assert qweight.dtype == torch.int32 and qweight.stride(0) == 1
    assert qzeros.tolist() == [8]


def test_scatter_and_gather_return_global_ids_with_excluded_tokens_masked():
    # The local winner on rank 1 must retain global ID 28, not local ID 12.
    state = head.DraftHeadInt4State(
        torch.empty(16, 8, dtype=torch.int32),
        torch.empty(1, 8, dtype=torch.float16),
        torch.tensor([8], dtype=torch.int8),
        128,
        torch.tensor([12, 3, 14]),
        16,
        32,
        {},
    )
    rank0 = torch.full((1, 16), -torch.inf, dtype=torch.float16)
    rank0[0, 3] = 2
    processor = SimpleNamespace(
        _gather_logits=lambda local: torch.cat([rank0, local], -1)
    )
    model = SimpleNamespace(_b70_lmhead_int4=state, logits_processor=processor)
    # Padded duplicate outputs are deliberately huge and must be discarded.
    local_output = torch.tensor(
        [[5, 1, 3, 100, 100, 100, 100, 100]], dtype=torch.float16
    )
    with patch.object(head, "int4_lmhead_logits", return_value=local_output):
        logits = head.draft_lmhead_int4_logits(model, torch.ones(1, 128).half())
    assert logits.shape == (1, 32)
    assert logits.argmax(-1).tolist() == [28]
    assert set(torch.isfinite(logits[0]).nonzero().flatten().tolist()) == {
        3,
        19,
        28,
        30,
    }


def test_empty_shard_participates_in_gather_without_launching_int4():
    state = head.DraftHeadInt4State(
        torch.empty(16, 0, dtype=torch.int32),
        torch.empty(1, 0).half(),
        torch.tensor([8], dtype=torch.int8),
        128,
        torch.empty(0, dtype=torch.long),
        16,
        32,
        {},
    )
    rank0 = torch.full((1, 16), -torch.inf, dtype=torch.float16)
    rank0[0, 7] = 4
    processor = SimpleNamespace(
        _gather_logits=lambda local: torch.cat([rank0, local], -1)
    )
    model = SimpleNamespace(_b70_lmhead_int4=state, logits_processor=processor)
    with patch.object(
        head, "int4_lmhead_logits", side_effect=AssertionError("empty shard")
    ):
        logits = head.draft_lmhead_int4_logits(model, torch.zeros(1, 128).half())
    assert logits.argmax(-1).tolist() == [7]
    assert torch.isneginf(logits[..., 16:]).all()
