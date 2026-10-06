# Compositional vocabulary pretraining

Train baseline and compositional language models with the English GPT-2 vocabulary or a Spanish 32k BPE vocabulary. The commands below cover tokenizer construction, preprocessing, training, and checkpoint loading. See the [method guide](../docs/method.md) for the model formulation and the [reproduction guide](../docs/reproduction.md) for experimental settings and reproducibility notes.

## Data and runtime requirements

From the repository root, run `uv sync --locked --all-extras --python 3.12`,
then `cd pretraining`. Use a PyTorch build matching the available CUDA GPUs
for training. W&B logging is disabled by default; enable it with
`--wandb_project YOUR_PROJECT` and optionally `--wandb_entity YOUR_ENTITY`.

The tokenizer builder needs UniMorph inflection TSV data for English and
Spanish. Set `UNIMORPH_ROOT` to the parent directory containing `eng/eng` and
`spa/spa` (a `.tsv` suffix is also accepted). If unset, the code looks in
`resources/unimorph/` at the repository root. Obtain it from the [UniMorph project](https://unimorph.github.io/) and retain
its data attribution and terms when redistributing it. English morphology
also uses `en_core_web_sm`, the NLTK WordNet corpus, PyEnchant, and an `en_US`
system dictionary.
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

Tokenize baseline data with the same FineWeb split and the matching tokenizer:

```bash
python tokenize_dataset.py --tokenizer openai-community/gpt2 \
  --dataset HuggingFaceFW/fineweb --dataset_config sample-10BT --max_tokens 1B

python tokenize_dataset.py \
  --tokenizer tokenizers/fineweb-2__spa_Latn__bpe32000__base_gpt2__stream__10Bbytes \
  --dataset HuggingFaceFW/fineweb-2 --dataset_config spa_Latn --max_tokens 1B
```

## Compositional tokenization flags

For English, restrict surface forms to the original vocabulary with `--skip_multi_token_words` in
both preprocessing and training. For Spanish, allow out-of-vocabulary compositions:
leave that flag off in both commands. Both languages also need `--skip_multi_token_bases`; it avoids unsupported
multi-token bases while still allowing Spanish multi-token surface forms.
Repeat the same flags and `--language` value for tokenization and training so
both commands load the same tokenizer bundle and dataset directory.

Transformation heads condition on the selected base's unembedding vector. The commands below set this with `--transformation_base_rep_source unembedding`.

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

With 49,152 tokens per rank per step, four ranks, and 5,000 iterations, these example commands process 983,040,000 training tokens (with gradient accumulation set to one).

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

## Cut cross entropy

`cut_cross_entropy/` is vendored from [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy). Its [license](cut_cross_entropy/LICENSE) and [acknowledgements](cut_cross_entropy/ACKNOWLEDGEMENTS.md) are kept in that folder; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

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
exported tokenizer. Training and native attention require CUDA.
