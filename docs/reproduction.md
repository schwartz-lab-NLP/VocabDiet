# Reproduction status

The [published paper](https://aclanthology.org/2026.findings-acl.1618/) and final manuscript define the experiment matrix. The workflow guides describe the released implementation and supported command-line arguments.

| Experiment | Paper setup | Artifact availability |
| --- | --- | --- |
| English representation probes | Llama-3-8B, Qwen2.5-7B, OLMo-2-7B; input composition and first ten layers; IV/OOV | Probe implementation included; complete original model-specific outputs are not bundled |
| Multilingual probes | ALLaM-7B (Arabic), EuroLLM-9B (German, Russian, Spanish) | Implementation included; obtain language-specific UniMorph resources |
| English post-hoc adaptation | Llama-3.1-8B, Qwen2.5-7B, OLMo-2-7B | Adaptation and downstream evaluation implementation included; checkpoints are external |
| Multilingual post-hoc adaptation | ALLaM-7B and EuroLLM-9B | Adaptation and evaluation implementation included; checkpoints are external |
| Pretraining | Baseline/compositional 124M models; English GPT-2 vocabulary, Spanish 32k BPE; about 1B tokens | Training implementation included; original final invocation records and checkpoints have not been recovered |
| Reallocation | Llama-3.1-8B; 10k freed slots and 2.5k new tokens per target language | Release implementation builds one shared multilingual tokenizer with disjoint language budgets; exact report underlying the published table has not been recovered |

## Paper hyperparameters

Post-hoc adaptation uses sequence length 256, 20k examples for one epoch, learning rate `5e-5`, warmup ratio `0.03`, and zero weight decay. Input and output transformation initialization uses single-transformation IV pairs. The main downstream setup applies detokenization filtering and excludes derivations. LoRA applies to the last eight blocks with `r = alpha = 256`.

English downstream benchmarks are MMLU, ARC, HellaSwag, Winogrande, TriviaQA, SQuAD, BoolQ, PIQA, and COPA. Multilingual benchmarks are XNLI, XQuAD, and Global MMLU. The paper uses five-shot evaluation and at most 5,000 examples per task.

Pretraining runs use four L40S GPUs. The example 5,000-step configuration with 49,152 tokens per rank consumes **983,040,000** training tokens. It is not an archived final run specification. In particular, verify the exact Spanish tokenizer, base-conditioned head settings, architecture, and validation text against the original training run before claiming a numerical reproduction.

## Reported values

These are transcribed from the final manuscript; they are not outputs of the release tests.

| Language | Vocabulary reduction | Baseline BPB | Compositional BPB | Baseline bytes/token | Compositional bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 41.6% | 1.08 | 1.09 | — | — |
| Spanish | 41.8% | 1.00 | 1.11 | 4.77 | 4.92 |

| Reallocation language | Baseline bytes/token | Reallocated bytes/token |
| --- | ---: | ---: |
| Arabic | 4.62 | 5.46 |
| Russian | 5.59 | 5.85 |
| German | 3.59 | 3.86 |
| Spanish | 3.80 | 4.07 |
| Macro average | 4.40 | 4.81 |

## Recording new runs

Keep the command, serialized arguments, random seed, dependency versions, model revision, morphology-resource revisions, tokenizer files, and the held-out document selection with each run. Keep raw output outside the source tree. CPU tests validate local components; GPU training and downstream performance remain separate validation steps.

## Architecture provenance

The manuscript calls the pretraining models `nanoGPT-124M`. The recovered
trainer defaults (12 blocks, hidden size 768, six attention heads, two KV heads,
gated intermediate size 2,048, untied embeddings) have **152,764,417** parameters
with the padded English vocabulary. These defaults are not established as the
published architecture. The implementation logs actual parameter counts and
saves constructor metadata. Obtain the final run configuration before selecting
architecture settings or presenting a new run as a reproduction of the 124M
experiments.

The publication code corrects padded-vocabulary scoring, tokenizer-specific EOS
boundaries, export attention/cache handling, and UTF-8 byte accounting. These
fixes change behavior relative to the recovered development files; historical
scores must be verified against the archived run rather than assumed unchanged.

Benchmark sampling in this release enforces a total 5,000-example cap across
MMLU subjects and records selected indices. Adaptation recomputes benchmark
scores on each invocation so a new training run cannot reuse stale scores
from the same output directory. The selected indices and full GPU scores
still need comparison with the original run artifacts.
