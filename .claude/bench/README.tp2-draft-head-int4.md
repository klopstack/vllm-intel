# TP2 reduced-vocabulary INT4 draft head experiment

This branch adds an opt-in Qwen3.5 MTP draft-head path for the B70 TP2 setup.
It selects a fixed global vocabulary, quantizes each rank's selected draft-head
rows to groupwise INT4, computes local logits, fills excluded rows with negative
infinity, and uses the existing logits processor to gather the full vocabulary.

The vocabulary file must contain one unique global token ID per line. The
validated experiment used 40,960 tokens and two XPU ranks. Run the CPU checks
from the repository root with:

```bash
.venv/bin/python -m pytest .claude/bench/test_tp2_draft_head.py -v
```

Enable the runtime path by adding these variables to the existing TP2 launch:

```bash
B70_DRAFT_LMHEAD_INT4=1 \
B70_DRAFT_VOCAB_PATH=/absolute/path/to/vocab-40960.txt \
VLLM_XPU_TRITON_ALLREDUCE=1 \
<existing B70 TP2 launch with use_local_argmax_reduction=false>
```

This is an experimental measurement path. Only the helper draft head and its
40K vocabulary cut use INT4; the target head and MTP body remain FP16. The
cross-rank all-gather stays outside the local-head graph. Synthetic two-rank
checks and one full-model run passed during development, but global output
equivalence has not been established. The path also rejects added vocabulary,
scaled or capped logits, non-FP16 head weights, `logits_as_input`, and local
argmax reduction.
