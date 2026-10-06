# Compositional vocabulary pretraining

This directory contains the recovered baseline and compositional training paths for
the English GPT-2 and Spanish 32k pretraining experiments described in the
paper. The paper states that both models train on FineWeb/FineWeb-2 for about
1B tokens, that English uses the GPT-2 vocabulary with OOV compositions
disabled, and that Spanish uses a 32k BPE vocabulary with OOV compositions
enabled. It does not give a complete command line or the full optimizer and
batch configuration.

## Data and runtime requirements

From the repository root, run `uv sync --locked --all-extras --python 3.12`,
then `cd pretraining`. Use a PyTorch build matching the available CUDA GPUs
for training. W&B logging is disabled by default; enable it with
`--wandb_project YOUR_PROJECT` and optionally `--wandb_entity YOUR_ENTITY`. Command-line help and module imports do not initialize CUDA.

The tokenizer builder needs UniMorph inflection TSV data for English and
Spanish. Set `UNIMORPH_ROOT` to the parent directory containing `eng/eng` and
`spa/spa` (a `.tsv` suffix is also accepted). If unset, the code looks in
`resources/unimorph/` at the repository root. No UniMorph data is bundled here;
obtain it from the [UniMorph project](https://unimorph.github.io/) and retain
its data attribution and terms when redistributing it. English morphology
also uses `en_core_web_sm`, the NLTK WordNet corpus, PyEnchant, and an `en_US`
system dictionary. Missing resources produce an actionable error when the
English morphology code first needs them.

Example setup:

```bash
export UNIMORPH_ROOT=/path/to/unimorph
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
```

## Tokenizers and data

The Spanish tokenizer command matching the paper's stated 10B-byte corpus and
32k vocabulary is:

```bash
python train_bpe_tokenizer.py \
  --dataset HuggingFaceFW/fineweb-2 \
  --dataset_config spa_Latn \
  --text_field text \
  --max_bytes 10B \
  --vocab_size 32000 \
  --output_dir tokenizers \
  --base_tokenizer openai-community/gpt2 \
  --streaming
```

This writes the tokenizer under
`tokenizers/fineweb-2__spa_Latn__bpe32000__base_gpt2__stream__10Bbytes`.
The older `EXPERIMENT_COMMANDS.md` leaves `--vocab_size` commented out in its
10B-token example; that would use GPT-2's 50,257-entry size, not the paper's
32k vocabulary. It also uses `--max_tokens` in that example, while the paper
specifies 10B **bytes**.

Tokenize baseline data with the same FineWeb split and the matching tokenizer:

```bash
python tokenize_dataset.py --tokenizer openai-community/gpt2 \
  --dataset HuggingFaceFW/fineweb --dataset_config sample-10BT --max_tokens 1B

python tokenize_dataset.py \
  --tokenizer tokenizers/fineweb-2__spa_Latn__bpe32000__base_gpt2__stream__10Bbytes \
  --dataset HuggingFaceFW/fineweb-2 --dataset_config spa_Latn --max_tokens 1B
```

## Compositional tokenization flags

The paper's English condition is IV-only: use `--skip_multi_token_words` in
both preprocessing and training. The Spanish condition allows OOV compositions:
leave that flag off in both commands. In the current single-ID implementation,
both languages also need `--skip_multi_token_bases`; it avoids unsupported
multi-token bases while still allowing Spanish multi-token surface forms.
Repeat the same flags and `--language` value for tokenization and training so
both commands load the same tokenizer bundle and dataset directory.
The paper's transformation head conditions on the selected base's output
unembedding vector; this is the default in the publication path. Pass
`--transformation_base_rep_source unembedding` explicitly if you want the
command itself to record that choice.

English compositional data and model:

```bash
python tokenize_compositional_dataset.py \
  --base_tokenizer openai-community/gpt2 --language en \
  --dataset HuggingFaceFW/fineweb --dataset_config sample-10BT \
  --max_tokens 1B --skip_multi_token_words --skip_multi_token_bases

torchrun --standalone --nproc_per_node=4 train_compositional.py \
  --base_tokenizer openai-community/gpt2 --language en \
  --dataset HuggingFaceFW/fineweb --dataset_config sample-10BT \
  --skip_multi_token_words --skip_multi_token_bases \
  --transformation_base_rep_source unembedding --num_iterations 5000
```

Spanish compositional data and model:

```bash
ES_TOKENIZER=tokenizers/fineweb-2__spa_Latn__bpe32000__base_gpt2__stream__10Bbytes

python tokenize_compositional_dataset.py \
  --base_tokenizer "$ES_TOKENIZER" --language es \
  --dataset HuggingFaceFW/fineweb-2 --dataset_config spa_Latn \
  --max_tokens 1B --skip_multi_token_bases

torchrun --standalone --nproc_per_node=4 train_compositional.py \
  --base_tokenizer "$ES_TOKENIZER" --language es \
  --dataset HuggingFaceFW/fineweb-2 --dataset_config spa_Latn \
  --skip_multi_token_bases --transformation_base_rep_source unembedding \
  --num_iterations 5000
```

These are example commands for a roughly 1B-token budget. With sequence length
49,152, four workers and 5,000 iterations consume 983,040,000 training tokens.
Exact final architecture, optimizer settings and invocation records must be
recovered before claiming a numerical reproduction of the published BPB.

Tokenized binary files are written under `data/<dataset>_<config>/<tokenizer>/`
for baseline runs and `data/vocab_diet/<dataset>_<config>/<bundle-cache-key>/`
for compositional runs. The preprocessor reserves validation documents from the
stream. Keep a shared held-out text split when comparing baseline and
compositional BPB.

## Lightweight CPU checks

These checks do not launch training or require a GPU:

```bash
python -m py_compile train_gpt.py train_compositional.py tokenize_dataset.py \
  tokenize_compositional_dataset.py decomposition_utils.py modeling_gpt.py \
  modeling_compositional.py
python train_gpt.py --help
python train_compositional.py --help
python tokenize_dataset.py --help
python tokenize_compositional_dataset.py --help
```

## Cut cross entropy provenance

`cut_cross_entropy/` is vendored from Apple's `apple/ml-cross-entropy` project,
The release retains the standard loss kernels used by factorized pretraining.
The retained upstream implementation identifies itself as version `25.7.2`;
the original vendored commit was not recorded, so this should not be presented
as a pinned upstream revision. Keep Apple's `LICENSE` and
`ACKNOWLEDGEMENTS.md` with the source. Apple's license grants use and
redistribution subject to its terms and disclaimers; it is not the repository's
Apache 2.0 license. The upstream project's current [license](https://github.com/apple/ml-cross-entropy/blob/main/LICENSE)
and [acknowledgements](https://github.com/apple/ml-cross-entropy/blob/main/ACKNOWLEDGEMENTS.md)
document the provenance and third-party notices. The removed `default/`,
`logit_scaling/`, `softcap_d/`, and Transformers integration subpackages were
not imported by the retained baseline or compositional training paths.

## Native checkpoints

Final checkpoints record effective arguments, exact model constructor settings,
weights, optimizer state, and compositional vocabulary mappings. Compiled-model
wrappers are removed from saved state keys. Baseline runs also export a
`GPTForCausalLM` checkpoint and tokenizer under `hf/`. Compositional runs save
their tokenizer under `tokenizer/` and retain native factorized scoring; they
are not exported as a baseline Hugging Face model.

From `pretraining/`:

```python
import torch
from checkpoint_utils import restore_model

checkpoint = torch.load("path/to/checkpoint.pt", map_location="cpu", weights_only=True)
model = restore_model(checkpoint, device="cuda")
```

Native inference takes extended token IDs, optional targets for teacher-forced
likelihood, and a sliding-window size in 128-token blocks. Use the matching
exported tokenizer. Training and native attention require CUDA; CPU tests cover
the component calculations and checkpoint metadata. The default
English architecture has 152,764,417 parameters, whereas the paper specifies
124M. See [architecture provenance](../docs/reproduction.md) before choosing a
configuration for a numerical reproduction.
