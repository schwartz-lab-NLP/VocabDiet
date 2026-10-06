# Post-hoc vocabulary adaptation and Patchscopes

Run these commands from the repository root after installation and the
[resource setup](../resources/README.md). Adaptation and full probes require a
CUDA GPU and access to the selected Hugging Face model. The supported output
model architectures are Llama, Qwen2, and OLMo2.

## Adapt an English model

```bash
python -m posthoc.adapt \
  --model_name meta-llama/Llama-3.1-8B \
  --unimorph_root "$UNIMORPH_ROOT" \
  --output_dir outputs/adaptation \
  --do_train --num_train_epochs 1 \
  --learning_rate 5e-5 --warmup_ratio 0.03 --weight_decay 0 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --bf16 --eval_ft --eval_downstream
```

The defaults initialize transformation vectors from single-transformation IV
pairs, filter failed detokenization, exclude derivations, and train on 20,000
FineWeb-Edu examples of length 256. Input offsets are distilled from the original
model first. The second stage freezes those input offsets and trains output
offsets together with LoRA adapters in the final eight blocks (`r = alpha = 256`).
Both stages use the same fixed sample. `--no-lora_ft` runs the transformation-only
condition. Batch size and accumulation above are example memory settings;
complete original invocation records are not bundled.

The paper also uses `Qwen/Qwen2.5-7B` and `allenai/OLMo-2-1124-7B` for English
adaptation. Use `--model_name` to choose the model. Benchmark defaults are the
full English tasks, five shots, and at most 5,000 examples per evaluator task.
For grouped benchmarks such as MMLU, the 5,000-example budget is shared across
subjects in proportion to their sizes, with at least one example per subject.
Selection uses seed 42 and is saved in `evaluation_samples.json`. The archived
sample indices are unavailable, so this implements the stated budget without
claiming the original exact sample selection. No tiny benchmark substitutes are used.

## Adapt a multilingual model

The multilingual entry point uses the same initialization, two-stage training,
and checkpoint implementation:

```bash
python -m posthoc.adapt_multilingual \
  --language spanish --model_name utter-project/EuroLLM-9B \
  --unimorph_root "$UNIMORPH_ROOT" \
  --output_dir outputs/adaptation \
  --do_train --num_train_epochs 1 \
  --learning_rate 5e-5 --warmup_ratio 0.03 --weight_decay 0 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --bf16 --eval_ft --eval_downstream
```

Choose `german` or `russian` with EuroLLM-9B, or `arabic` with
`ALLaM-AI/ALLaM-7B-Instruction-preview`. Supply the corresponding UniMorph data.
The default training corpus remains FineWeb-Edu, as described in the paper.
`--dataset_language` controls language selection for datasets that support it.
Multilingual benchmarks are XNLI, XQuAD, and Global MMLU. The evaluator reports
unavailable language/task combinations; available tasks use five shots.

## Inspect composed input representations

The English Patchscopes experiments include derivations. This example uses the
paper's Llama-3-8B input probe model:

```bash
python -m posthoc.patchscopes \
  --model_name meta-llama/Meta-Llama-3-8B \
  --unimorph_root "$UNIMORPH_ROOT" \
  --include_derivations --last_analysis_layer 10 \
  --output_dir outputs/patchscopes
```

For multilingual probes:

```bash
python -m posthoc.patchscopes_multilingual \
  --language spanish --model_name utter-project/EuroLLM-9B \
  --unimorph_root "$UNIMORPH_ROOT" \
  --last_analysis_layer 10 --output_dir outputs/patchscopes
```

Outputs include `patchscopes_resolution_layer.csv`, transformation summaries,
and cached probe results. Layer zero represents the embedding; detokenization
success is a match within layers zero through ten. See the
[analysis guide](../analysis/README.md) for aggregating IV/OOV results. Probe
caches and adaptation outputs are scoped by model and language. Cached pickle
files are local intermediate outputs; use only caches you generated yourself.

## Saved models and evaluation results

Each adaptation run saves `config.json`, `training_config.json`, metrics,
`removed_vocab_inflection_tokens.json`, and a `checkpoint/` directory beneath
its model/language/settings output directory. The removed-token list is the
input to the [joint vocabulary reallocation analysis](../analysis/README.md).

Reload an exported checkpoint without downloading the original model:

```python
from posthoc.checkpoint import load_adaptation

model, tokenizer = load_adaptation("path/to/checkpoint", device="cuda")
inputs = tokenizer("They walked", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.batch_decode(output, skip_special_tokens=True))
```

The inference export includes model weights, offsets, vocabulary mappings, and
both tokenizer definitions. LoRA adapters are merged at the end of the run.
This is a local inference format; optimizer state and resuming training are not
part of its contract. Use `load_adaptation`, since a bare Transformers model
loader does not restore the additive vocabulary mappings.

`--help` lists workflow options. Standard Hugging Face Trainer arguments are
accepted by the adaptation entry point; unknown arguments are rejected.
The [reproduction guide](../docs/reproduction.md) distinguishes tested code
contracts from full GPU runs and archived numerical results.
