# Analysis workflows

Aggregate experiment outputs into CSV summaries or reallocate vocabulary slots across languages. Run commands from the repository root after installing the [analysis dependencies](../README.md#installation). See the [reproduction guide](../docs/reproduction.md) for reproducibility notes.

## Patchscopes resolution and failure analysis

Create a manifest CSV with one row per model/language run and columns `model,language,path`. `path` must point to that run's `patchscopes_resolution_layer.csv`:

```csv
model,language,path
llama3,en,/path/to/patchscopes_resolution_layer.csv
```

Run:

```bash
python analysis/aggregate_patchscopes.py \
  --manifest patchscopes_manifest.csv \
  --output-dir analysis-output \
  --figure
```

The output includes per-transformation and per-class CSVs, plus an exploratory exemplar-count correlation CSV. By default it keeps rows marked `is_single_type`, matching the paper's single-transformation analysis. IV means the target has one original model token (`word_n_tokens == 1`); OOV means it has more than one. `embed_success` is the share whose additive representation first matches at layer zero (`first_layer_additive == 0`). `detok_success` is the share with a recorded match within layers zero through ten (`0 <= first_layer_additive <= 10`). The optional figure plots additive embed success for IV and OOV targets by coarse class. `--all-types` includes rows outside the single-transformation subset.

The script groups transformation labels as follows: labels with `;` are inflections, affix-like labels (including `-less`, `pre-`, `un-`) are derivations, and `add_capitalization` is capitalization. Unrecognized/base rows are left out of class summaries.

The exemplar-count CSV reports Spearman correlations between `log(IV target count)` and additive success across transformations. Each correlation includes transformations with at least one IV target; the OOV correlation also requires at least one OOV target.

## Downstream evaluations

Create a manifest with the same `model,language,path` columns, where `path` points to an evaluator `downstream_metrics.json`:

```csv
model,language,path
llama3,en,/path/to/downstream_metrics.json
```

Run:

```bash
python analysis/aggregate_downstream.py \
  --manifest downstream_manifest.csv \
  --output downstream_tidy.csv
```

The script flattens the saved task → method → metric structure, including nested evaluator task groups. It preserves task, method, and metric names from the JSON and writes one numeric value per row; it does not infer task-specific display names or rescale scores.

## Joint vocabulary reallocation analysis

`reallocate_vocabulary.py` constructs one shared multilingual tokenizer. It trains a candidate BPE tokenizer per language, selects distinct reachable tokens in rank order, and allocates a fixed budget per language into disjoint freed IDs. The same resulting tokenizer is evaluated on every language's held-out text. For four languages and the default 2,500 slots per language, the tokenizer has 10,000 reallocated slots total.

Example invocation (the removed-token file and datasets must be available locally or through Hugging Face Datasets):

```bash
python -m analysis.reallocate_vocabulary \
  --base_tokenizer meta-llama/Llama-3.1-8B \
  --removed_tokens_json outputs/adaptation/removed_vocab_inflection_tokens.json \
  --languages ar ru de es \
  --tokens_per_language 2500 \
  --output_root outputs/reallocation
```

The run writes a single tokenizer under `tokenizer/`, an `allocation.json` assigning each language's tokens to freed IDs, and `report.json`, `language_metrics.csv`, and `report.md`. `report.json` records data limits, token counts, per-language measurements, and the joint allocation mode.

The CSV aggregation helpers require pandas, numpy, and scipy; plot generation also requires matplotlib (`pip install -e '.[analysis]'`). The vocabulary reallocation helper additionally uses the project's datasets, tokenizers, tqdm, and transformers dependencies.
