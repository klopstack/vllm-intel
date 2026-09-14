# Intel B70 performance research — 2026-09-14

This backup preserves the benchmark results, traces, harnesses, source snapshots,
and rejected experiments from the local investigation. It includes earlier
benchmark history for comparison. AI assistance was used for implementation,
experiments, and analysis.

## Code branches

- [xpu-fp16-allreduce](https://github.com/CySpiegel/vllm-intel/tree/xpu-fp16-allreduce):
  FP16 support in the fork's existing opt-in custom TP2 all-reduce.
- [xpu-tp2-draft-head-int4](https://github.com/CySpiegel/vllm-intel/tree/xpu-tp2-draft-head-int4):
  separate experimental INT4 draft head and 40960-token draft vocabulary.
- [xpu-tp1-draft-int4-experiment](https://github.com/CySpiegel/vllm-intel/tree/xpu-tp1-draft-int4-experiment):
  earlier single-GPU draft head/body experiments and their local overlays.
- [xpu-graph-capture-profiler](https://github.com/CySpiegel/vllm-intel/tree/xpu-graph-capture-profiler):
  device selection for the existing graph-capture profiler flag.

The user authorized pushing the work to their fork. These backups do not
authorize or create additional pull requests or promote experiments to defaults.
Historical artifacts retain the status and authorization statements recorded
when they were written; this backup supersedes their statements that no push was
authorized. Human approval remains required for any new PR.

## Measured model performance

Native decode tokens/s, excluding prefill; medians over five fixed prompts at
each context length, with MTP6. Both output heads remain unquantized in the
native-head rows. The target checkpoint body was already GPTQ INT4.

| Configuration | 512 tokens | 8192 tokens | 32768 tokens |
| --- | ---: | ---: | ---: |
| Historical TP1, native heads | 68.87 | 54.21 | 49.95 |
| TP2, native heads, oneCCL | 114.27 | 86.57 | 80.93 |
| TP2, native heads, custom all-reduce | 112.39 | 84.05 | 79.91 |
| TP2, native heads, oneCCL repeat | 114.24 | 86.46 | 83.73 |
| TP2, INT4 draft head + 40K cut, custom all-reduce | 133.82 | 109.82 | 95.03 |

The native-head TP2 configuration exceeded 100 tok/s at short context. The custom
all-reduce did not improve full-model throughput; it remains disabled by default.
The extra quantized point exceeded 100 tok/s at short and 8K context, with the
target head and MTP body unchanged. The TP1 comparison uses the earlier HTTP
harness; TP2 runs use the current offline harness.

The corpus is controlled repeated prose with fixed input token IDs, rather than
a broad evaluation of application tasks. Each request starts with a cleared
prefix cache. Runtime: custom Qwen3.8-27B, FP16, FP8 KV, XPU graphs, V1 runner,
FlashAttention, maximum length 98304, maximum sequences 1, batched tokens 6656,
memory utilization 0.93, greedy sampling, seed 20260914.

All four smoke prompts matched across the current runs. Exact timed-output
equality was not established: 8/15 off/on pairs differed, while repeating the
unchanged oneCCL baseline also changed 7/15 outputs. These are throughput
experiments with limited correctness checks, not a comprehensive accuracy eval.
[tp2-results.json](tp2-results.json) records per-prompt timing, acceptance,
source provenance, and the independent comparison.

## Validation already completed

- FP16 all-reduce: 9 CPU tests; on each B70, 20000 randomized calls, 200000
  mixed-dtype ordering calls, and 2000 mixed-dtype graph replays, with zero errors.
  Scoped pre-commit and commit checks passed.
- TP2 draft-head experiment: 8 CPU tests; both actual shard sizes and an empty
  shard passed dequantized FP32-reference, global token mapping, and local-head
  graph-replay checks. Native all-gather was outside that captured region.
  Capturing the entire isolated head plus collective failed and is preserved.
- Profiler fix: 34 tests passed, 33 CUDA-only tests skipped; isolated Intel model
  tests confirmed capture traces and matching outputs. NVIDIA/AMD hardware was
  not tested.

No GPU benchmarks were repeated for this backup. Failed setup runs, the failed
strict output comparison, and superseded kernel-harness checks are retained.
The exact tested source archives distinguish measured code from later formatting.

## Findings and remaining experiments

1. Test the existing `use_local_argmax_reduction=True` option with native heads.
   The baseline left it off. For greedy non-tree drafting, the proposer can use
   each shard's local maximum and gather only value/index pairs instead of full
   vocabulary logits. The local Qwen3.5 MTP model implements the required mixin.
   This retains FP16 heads and the full vocabulary; its TP2 speed and output
   behavior still need benchmarking.
2. Restrict the custom collective to sizes where it wins. Graph microbenchmarks
   favored it through 15360 elements and favored oneCCL from 20480 upward.
   A size-based policy still needs matched model benchmarks.
3. Make attention work scale safely with the live context under graph replay.
   The isolated short-context M7 kernel improved 36.4% with an actual-length
   bound, and the 8K case improved 4.1%. A static smaller capture bound is unsafe
   as context grows; no such product change was integrated.
4. Optimize the existing FP16 output projections and existing target GPTQ GEMMs.
   They dominated recorded device durations. The alternative layouts and Triton
   kernels tested so far were flat or slower. This leaves substantial work to
   investigate, without claiming an available speedup.
5. Sweep MTP depth on TP2 with unchanged heads, measuring accepted tokens per
   second across the fixed contexts and a broader task corpus.

Native M7-to-M8 padding gave a roughly 3% isolated down-projection improvement
while making gate/up slower. Generic attention was much slower. Recorded small
copies did not establish a large removable driver overhead. None of these
isolated results is presented as an end-to-end model improvement.

## Evidence archive

[evidence.tar.gz](evidence.tar.gz) preserves original bytes under `.claude/bench`,
`.claude/docs`, selected handoff/specification files, and `worktree-snapshots`.
[manifest.json](manifest.json) lists every member's size and SHA256 plus the
archive hash. Compiler caches, model weights, environments, symlinks, and Git
metadata are excluded. Use a new empty directory when extracting:

```sh
mkdir xpu-evidence
tar -xzf research/xpu-20260914/evidence.tar.gz -C xpu-evidence
```

Useful extracted reports:

- `.claude/docs/xpu-fp16-tp2-benchmark-20260914.md`
- `.claude/docs/trace-guided-kernel-experiments-20260914.md`
- `.claude/docs/astra-full-vocab-evidence-20260914.md`
- `.claude/docs/xpu-capture-profiler-fix-20260914.md`
- `.claude/bench/LEDGER.md`

Harnesses retain their original machine paths and environment assumptions.
Review and adapt them before replaying an experiment. No persistent model server
was left running at the end of the recorded work.
