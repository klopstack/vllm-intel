# XPU TP1 draft INT4 experiment

This branch preserves exploratory runtime hooks that existed as uncommitted
overlays on source base `319cc5ef19946d34c2e66cbbec5bda29d0bfa328`. The hooks
quantize the Qwen3.5 MTP draft body and a separate copy of the shared FP16 LM
head weights to GPTQ INT4 group size 128 at runtime, route them through the XPU
`int4_gemm_w4a16` operator, and retain related XPU memory-query and
graph-profiler changes. The target LM head remains FP16.

The experiment is opt-in through `B70_DRAFT_MTP_INT4=1` and
`B70_DRAFT_LMHEAD_INT4=1`. `B70_DRAFT_VOCAB_PATH` optionally selects a reduced
draft vocabulary. Each distinct flag mode requires a separate compiler cache.

Only tensor parallel size 1 was targeted. Unknown tensor parallel size and
tensor parallel size greater than 1 fail closed; other accelerators and model
families are unsupported. This is an archival experiment, not production-ready
functionality.

An earlier full-vocabulary body-and-head INT4 MTP6 run measured 79.69 tokens/s
at 8K versus 54.12 tokens/s for the native run. Its evidence is archived on the
main research branch in `research/xpu-20260914/evidence.tar.gz`; its historical
path inside that archive is
`.claude/bench/results/astra-full-vocab-20260914`. This branch adds no new
numerical validation, and model-level equivalence has not been established.
