# Reproduction guide

Use the workflow guides to run the experiments. The [paper](https://aclanthology.org/2026.findings-acl.1618/) defines the experimental comparisons; the settings below summarize the camera-ready version.

## Experiments

| Experiment | Setup | Guide |
| --- | --- | --- |
| English representation probes | Llama-3.1-8B, Qwen2.5-7B, and OLMo-2-7B; embedding layer and layers 1–10; in-vocabulary and out-of-vocabulary words | [Patchscopes](../posthoc/README.md#inspect-composed-input-representations) |
| Multilingual probes | ALLaM-7B for Arabic; EuroLLM-9B for German, Russian, and Spanish | [Patchscopes](../posthoc/README.md#inspect-composed-input-representations) |
| English post-hoc adaptation | Llama-3.1-8B, Qwen2.5-7B, and OLMo-2-7B | [Adaptation](../posthoc/README.md#adapt-an-english-model) |
| Multilingual post-hoc adaptation | ALLaM-7B for Arabic; EuroLLM-9B for German, Russian, and Spanish | [Adaptation](../posthoc/README.md#adapt-a-multilingual-model) |
| Pretraining | Baseline and compositional nanoGPT-124M models; GPT-2 vocabulary for English, 32k BPE for Spanish; about 1B training tokens per model | [Pretraining](../pretraining/README.md) |
| Vocabulary reallocation | Llama-3.1-8B tokenizer; 10,000 freed English slots, with 2,500 new tokens each for Arabic, Russian, German, and Spanish | [Reallocation](../analysis/README.md#joint-vocabulary-reallocation-analysis) |

## Experimental settings

Post-hoc adaptation trains on 20,000 FineWeb-Edu examples for one epoch, with sequence length 256, learning rate `5e-5`, warmup ratio `0.03`, and zero weight decay. Transformation vectors are initialized from in-vocabulary pairs with one transformation. The downstream setup filters failed detokenization and excludes derivations. LoRA applies to the last eight blocks with `r = alpha = 256`.

English downstream benchmarks are MMLU, ARC, HellaSwag, Winogrande, TriviaQA, SQuAD, BoolQ, PIQA, and COPA. Multilingual benchmarks are XNLI, XQuAD, and Global MMLU. The camera-ready appendix specifies five in-context examples per task and up to 5,000 evaluation examples per dataset.

Pretraining uses FineWeb for English and FineWeb-2 for Spanish. English restricts compositions to the original GPT-2 vocabulary. Spanish uses a 32k BPE vocabulary trained on 10B bytes of Spanish FineWeb-2 and permits out-of-vocabulary compositions.

## Reported values

The tables below report the paper's baseline and compositional pretraining results and its tokenizer reallocation results. Lower bits per byte (BPB) is better; higher bytes per token indicates more efficient tokenization.

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

## Reproducibility notes

The repository includes experiment code but not the paper's trained checkpoints, raw results, exact pretraining and reallocation run configurations, or original benchmark sample indices. Fresh runs may differ from the reported values.

The paper specifies nanoGPT-124M, while the default English configuration here has 152,764,417 parameters.

The exemplar-count correlation helper provides exploratory summaries of supplied probe outputs. Reproducing the paper's Spearman values requires the same transformation selection and aggregation. Offset-geometry and vocabulary-scaling results also require raw measurements beyond the supplied workflows.

CPU tests check component behavior; they do not reproduce the paper's GPU training or benchmark scores.

## Recording new runs

Save the command, arguments, seed, dependency versions, model and morphology-resource revisions, tokenizer files, and held-out document selection with each run. The evaluator writes its selected benchmark indices to `evaluation_samples.json`. Keep raw results outside the source tree.
