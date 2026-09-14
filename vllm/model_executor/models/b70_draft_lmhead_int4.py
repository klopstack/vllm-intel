# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""B70 Phase S runtime helper: draft MTP LM head INT4 g128 sym.

Escribido por patch_draft_lmhead_int4.py dentro del contenedor. Provee la
cuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el
ruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda
fp16 (lossless).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles
               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])
      scales:  fp16 [K//group_size, N]
      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16
    """
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, chunk):
        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)  # [c, g]
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8  # 0..15
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(wc.shape[0], K // 8)
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32
    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16
    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas
    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)
    scales = scales_contig.t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    """Cuantiza el lm_head fp16 compartido del draft (one-time, no-op si no
    hay env gate o si ya se construyo). Almacena en model._b70_lmhead_int4."""
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_b70_lmhead_int4_tp_blocked", False):
        return
    parallel_config = getattr(
        getattr(model, "vllm_config", None), "parallel_config", None
    )
    tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tp_size is None:
        model._b70_lmhead_int4_tp_blocked = True
        print(
            "[B70] draft LM head INT4: TP unknown; skipping (fail-closed, issue #9)",
            flush=True,
        )
        return
    if tp_size > 1:
        model._b70_lmhead_int4_tp_blocked = True
        print(
            f"[B70] draft LM head INT4: TP>1 (tp={tp_size}) detected; skipping "
            "(C1/TP1 only, issue #9)",
            flush=True,
        )
        return
    if getattr(model, "_b70_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        print(
            "[B70] draft LM head INT4: lm_head.weight no disponible; "
            "draft sigue por fp16",
            flush=True,
        )
        return
    vocab_path = os.environ.get("B70_DRAFT_VOCAB_PATH")
    token_ids = None
    quant_weight = weight.detach()
    if vocab_path:
        try:
            ids = [
                int(line)
                for line in Path(vocab_path).read_text().splitlines()
                if line.strip()
            ]
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"invalid B70 draft vocabulary {vocab_path}: {exc}"
            ) from exc
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("B70 draft vocabulary must be non-empty and unique")
        if min(ids) < 0 or max(ids) >= weight.shape[0]:
            raise RuntimeError("B70 draft vocabulary contains an out-of-range token ID")
        if len(ids) % 8:
            raise RuntimeError("B70 draft vocabulary size must be divisible by 8")
        token_ids = torch.tensor(ids, dtype=torch.long, device=weight.device)
        quant_weight = weight.detach().index_select(0, token_ids)
        print(
            f"[B70] reduced draft vocabulary: {len(ids)}/{weight.shape[0]} "
            f"tokens from {vocab_path}",
            flush=True,
        )
    print(
        "[B70] draft LM head INT4: cuantizando lm_head fp16 "
        f"{tuple(quant_weight.shape)} -> INT4 g128 sym (one-time)",
        flush=True,
    )
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(quant_weight)
    model._b70_lmhead_int4 = (
        qweight,
        scales,
        qzeros,
        group_size,
        token_ids,
        weight.shape[0],
    )
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + (
        scales.numel() * scales.element_size()
    )
    print(
        f"[B70] draft LM head INT4: listo. {fp16_bytes / 1e9:.2f} GB fp16 -> "
        f"{int4_bytes / 1e9:.2f} GB INT4 (ahorro "
        f"{(fp16_bytes - int4_bytes) / 1e6:.1f} MB/lectura)",
        flush=True,
    )


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size, token_ids, full_vocab_size = (
        model._b70_lmhead_int4
    )
    logits = int4_lmhead_logits(hidden_states, qweight, scales, qzeros, group_size)
    if token_ids is not None:
        reduced_logits = logits
        logits = torch.full(
            (*reduced_logits.shape[:-1], full_vocab_size),
            -torch.inf,
            dtype=reduced_logits.dtype,
            device=reduced_logits.device,
        )
        logits.index_copy_(-1, token_ids, reduced_logits)
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
