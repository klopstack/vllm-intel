# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed._symmetric_memory as symm

from vllm.distributed.device_communicators.xpu_triton_all_reduce import (
    BLOCK,
    MAX_NUMEL,
    OneShotAllReduce,
)


@pytest.fixture
def cpu_all_reduce():
    group = MagicMock()
    group.rank.return_value = 0
    group.size.return_value = 2
    group.group_name = "test"
    peer_slot = torch.empty(2 * MAX_NUMEL, dtype=torch.bfloat16)
    peer_flags = torch.empty(MAX_NUMEL // BLOCK, dtype=torch.int32)
    slot_handle = MagicMock()
    slot_handle.get_buffer.return_value = peer_slot
    flag_handle = MagicMock()
    flag_handle.get_buffer.return_value = peer_flags

    def empty(numel, *, dtype, device):
        return torch.empty(numel, dtype=dtype)

    with (
        patch.object(symm, "enable_symm_mem_for_group"),
        patch.object(symm, "empty", side_effect=empty),
        patch.object(symm, "rendezvous", side_effect=[slot_handle, flag_handle]),
        patch(
            "vllm.distributed.device_communicators.xpu_triton_all_reduce.dist.barrier"
        ),
    ):
        return OneShotAllReduce(group, torch.device("cpu"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_custom_all_reduce_accepts_supported_16_bit_dtypes(dtype):
    all_reduce = OneShotAllReduce.__new__(OneShotAllReduce)

    assert all_reduce.should_custom_ar(torch.empty(BLOCK, dtype=dtype))


@pytest.mark.parametrize(
    "tensor",
    [
        torch.empty(BLOCK, dtype=torch.float32),
        torch.empty(BLOCK + 1, dtype=torch.float16),
        torch.empty(MAX_NUMEL + BLOCK, dtype=torch.bfloat16),
        torch.empty((BLOCK, 2), dtype=torch.float16)[:, 0],
        torch.empty(0, dtype=torch.bfloat16),
    ],
)
def test_custom_all_reduce_rejects_unsupported_inputs(tensor):
    all_reduce = OneShotAllReduce.__new__(OneShotAllReduce)

    assert not all_reduce.should_custom_ar(tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_reduce_uses_initialized_matching_slot_view(cpu_all_reduce, dtype):
    all_reduce = cpu_all_reduce
    launch = MagicMock()
    kernel = MagicMock()
    kernel.__getitem__.return_value = launch
    tensor = torch.ones(BLOCK, dtype=dtype)

    with patch(
        "vllm.distributed.device_communicators.xpu_triton_all_reduce."
        "_one_shot_push_kernel",
        kernel,
    ):
        output = all_reduce.all_reduce(tensor)

    assert output.dtype == dtype
    args = launch.call_args.args
    assert args[2].dtype == dtype
    assert args[3].dtype == dtype
    assert args[2].untyped_storage().data_ptr() == all_reduce._slot.data_ptr()
    assert args[3].untyped_storage().data_ptr() == all_reduce._peer_slot.data_ptr()
    if dtype == torch.float16:
        value = torch.tensor([1.0009765625], dtype=torch.float16)
        args[2][:1].copy_(value)
        assert torch.equal(args[2][:1], value)
        assert all_reduce._slot[:1].item() != value.item()
