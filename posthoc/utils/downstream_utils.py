import numpy as np
import torch
import os
import json
import hashlib
from collections import defaultdict

ENGLISH_TASKS = (
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "triviaqa",
    "squadv2",
    "boolq",
    "piqa",
    "copa",
)

SAVE_PREDS_TO_FILE_TASKS = ["squadv2", "triviaqa", "xquad"]
GENERATIVE_TASKS = ["squadv2", "triviaqa", "xquad"]


task_to_limited_samples = defaultdict(lambda: 5000)

generation_tasks = []


def sample_benchmark(lengths, cap, seed=42):
    """Choose a deterministic total budget across a benchmark's leaf tasks."""
    names = sorted(name for name, length in lengths.items() if length > 0)
    if cap <= 0:
        raise ValueError("Evaluation sample cap must be positive")
    if cap < len(names):
        raise ValueError("Sample cap must cover at least one example per nonempty subtask")
    total = sum(lengths[name] for name in names)
    cap = min(cap, total)
    quotas = {name: lengths[name] for name in names}
    if cap < total:
        remainder = cap - len(names)
        available = total - len(names)
        raw = {name: remainder * (lengths[name] - 1) / available for name in names}
        quotas = {name: 1 + int(raw[name]) for name in names}
        extra = cap - sum(quotas.values())
        order = sorted(names, key=lambda name: (-(raw[name] - int(raw[name])), name))
        for name in order[:extra]:
            quotas[name] += 1
    rng = np.random.default_rng(seed)
    return {
        name: sorted(rng.choice(lengths[name], quotas[name], replace=False).tolist())
        for name in names
    }


def evaluation_selection(task_manager, task_name, cap=None):
    loaded = task_manager.load([task_name])
    leaves = loaded["tasks"]
    root = loaded["groups"].get(task_name) or leaves.get(task_name)
    selected_tasks = [root] if root is not None else list(leaves.values())
    samples = (
        sample_benchmark({name: len(task.eval_docs) for name, task in leaves.items()}, cap)
        if cap is not None
        else None
    )
    return selected_tasks, samples


def _get_cache_key(
    model_name, limited, examples_limit, task_names, get_inflection_metrics, cache_name
):
    """Generate a cache key based on evaluation parameters."""
    # Create a deterministic string from the parameters
    key_components = [
        "paper-five-shot-global-cap-v2",
        str(cache_name or "default"),  # Add cache_name to separate different eval types
        str(model_name),
        str(limited),
        str(examples_limit),
        str(sorted(task_names)) if task_names else "all_default_tasks",
        str(get_inflection_metrics),
    ]
    key_string = "|".join(key_components)

    # Generate hash from the key string
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    return cache_key


def _get_cache_path(cache_key):
    """Get the cache file path for a given cache key."""
    cache_dir = os.path.expanduser(
        os.environ.get("POSTHOC_CACHE_DIR", "~/.cache/vocab-diet/posthoc")
    )
    cache_dir = os.path.join(cache_dir, "downstream_eval")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.json")


def _load_cache(cache_key):
    """Load cached results if they exist."""
    cache_path = _get_cache_path(cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache is corrupted, ignore it
            pass
    return None


def _save_cache(cache_key, results):
    """Save results to cache."""
    cache_path = _get_cache_path(cache_key)
    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
    except IOError:
        # If we can't save to cache, just continue
        pass


def _get_per_run_cache_path(output_dir, stage_name, task_name):
    """Get the per-run cache file path for a specific stage and task."""
    cache_dir = os.path.join(output_dir, "eval_cache", "paper-five-shot-global-cap-v2", stage_name)
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{task_name}.json")


def _load_per_run_cache(output_dir, stage_name, task_name):
    """Load per-run cached results for a specific stage and task."""
    cache_path = _get_per_run_cache_path(output_dir, stage_name, task_name)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache is corrupted, ignore it
            pass
    return None


def _save_per_run_cache(output_dir, stage_name, task_name, results):
    """Save per-run cached results for a specific stage and task."""
    cache_path = _get_per_run_cache_path(output_dir, stage_name, task_name)
    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
    except IOError:
        # If we can't save to cache, just continue
        pass


def evaluate_model(
    model,
    tokenizer,
    limited=False,
    task_names=None,
    output_path=None,
    examples_limit=None,
    get_inflection_metrics=False,
    chat_format=False,
    use_cache=False,
    cache_name=None,
    use_per_run_cache=False,
    per_run_cache_dir=None,
    per_run_stage_name=None,
    overwrite_cache=False,
    overwrite_per_run_cache=False,
):
    try:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval import tasks
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "Downstream benchmark evaluation requires lm-eval. Install the posthoc optional dependencies."
        ) from exc
    """
    Evaluate model on multiple benchmarks using lm-evaluation-harness.
    """
    # Extract model name for caching
    model_name = getattr(model, "name_or_path", None) or getattr(
        model.config, "name_or_path", "unknown_model"
    )

    curr_task_to_limited_samples = task_to_limited_samples
    if examples_limit is not None:
        curr_task_to_limited_samples = defaultdict(lambda: examples_limit)

    task_configs = {name: {"num_fewshot": 5} for name in ENGLISH_TASKS}

    if task_names is not None:
        task_configs = {task_name: task_configs[task_name] for task_name in task_names}

    # Check cache before running evaluation (only if use_cache is True)
    if use_cache:
        task_list = list(task_configs.keys())
        cache_key = _get_cache_key(
            model_name, limited, examples_limit, task_list, get_inflection_metrics, cache_name
        )
        if not overwrite_cache:
            cached_results = _load_cache(cache_key)
            if cached_results is not None:
                cache_desc = f" ({cache_name})" if cache_name else ""
                print(
                    f"Using cached results for model {model_name}{cache_desc} with {len(task_list)} tasks"
                )
                return cached_results

    task_manager = tasks.TaskManager()
    model.eval()
    adapted_model = HFLM(
        model,
        tokenizer=tokenizer,
        trust_remote_code=True,
    )

    # Evaluate the model on each task
    results = {}
    for task_name, config in task_configs.items():
        # Check per-run cache for this specific task
        if (
            use_per_run_cache
            and per_run_cache_dir
            and per_run_stage_name
            and not overwrite_per_run_cache
        ):
            cached_task_result = _load_per_run_cache(
                per_run_cache_dir, per_run_stage_name, task_name
            )
            if cached_task_result is not None:
                print(f"Using cached result for task {task_name} in stage {per_run_stage_name}")
                results[task_name] = cached_task_result
                continue

        # Evaluate the model on the task
        curr_output_path = os.path.join(output_path, task_name) if output_path else None
        if curr_output_path:
            os.makedirs(curr_output_path, exist_ok=True)
        selected_tasks, samples = evaluation_selection(
            task_manager, task_name, curr_task_to_limited_samples[task_name] if limited else None
        )
        if curr_output_path and samples is not None:
            with open(os.path.join(curr_output_path, "evaluation_samples.json"), "w") as f:
                json.dump(samples, f, indent=2)
        with torch.no_grad():
            result = simple_evaluate(
                model=adapted_model,
                tasks=selected_tasks,
                samples=samples,
                task_manager=task_manager,
                batch_size=1,
                num_fewshot=config.get("num_fewshot", None),
                gen_kwargs=config.get("gen_kwargs", None),
                log_samples=True if curr_output_path else False,
            )

        # Store base results
        if len(result["results"].keys()) > 1:
            results[task_name] = dict()
            for subtask_name in result["results"].keys():
                if task_name.startswith("global_mmlu") and subtask_name.startswith(
                    f"global_mmlu_full_en_"
                ):
                    continue
                results[task_name][subtask_name] = result["results"][subtask_name]
        else:
            results[task_name] = result["results"][task_name]

        if get_inflection_metrics and task_name in GENERATIVE_TASKS:
            # Get metrics after task completion
            metrics = model.get_metrics_and_reset()
            reordered_metrics = dict()

            # Reorder metrics to match evaluation order
            preds = result["samples"][task_name]
            eval_inputs = []
            for pred in preds:
                if "arguments" in pred and pred["arguments"]:
                    eval_inputs.append(
                        pred["arguments"][0][0]
                        if isinstance(pred["arguments"], list)
                        else str(pred["arguments"])
                    )
                elif "doc" in pred:
                    eval_inputs.append(str(pred["doc"]))
                else:
                    eval_inputs.append("")

            # Create reorder indices
            reorder_indices = []
            matched_indices = []
            for orig_i, eval_input in enumerate(eval_inputs):
                for i, cached_input in enumerate(metrics["inputs"]):
                    if i not in reorder_indices and eval_input[-100:] == cached_input[-100:]:
                        reorder_indices.append(i)
                        matched_indices.append(orig_i)
                        break

            # Reorder all metric lists
            for key in ["input_inflection_counts", "output_inflection_counts", "inputs", "outputs"]:
                if key in metrics:
                    metrics[key] = [metrics[key][i] for i in reorder_indices]

            # Calculate inflection statistics
            total_instances = len(metrics["input_inflection_counts"])
            input_with_inflections = sum(
                1 for count in metrics["input_inflection_counts"] if count > 0
            )
            output_with_inflections = sum(
                1 for count in metrics["output_inflection_counts"] if count > 0
            )
            both_with_inflections = sum(
                1
                for i, o in zip(
                    metrics["input_inflection_counts"], metrics["output_inflection_counts"]
                )
                if i > 0 and o > 0
            )

            # Calculate token save rates
            def calculate_save_rate(inflection_token_counts, total_tokens):
                if total_tokens == 0:
                    return 0.0
                saved_tokens = sum(
                    count * (token_count - 1)
                    for token_count, count in inflection_token_counts.items()
                )
                return saved_tokens / total_tokens

            input_save_rate = calculate_save_rate(
                metrics["input_inflection_token_counts"], metrics["input_total_tokens"]
            )
            output_save_rate = calculate_save_rate(
                metrics["output_inflection_token_counts"], metrics["output_total_tokens"]
            )

            # Add inflection metrics to results
            inflection_metrics = {
                "input_inflection_proportion": input_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "output_inflection_proportion": output_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "both_inflection_proportion": both_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "input_token_save_rate": input_save_rate,
                "output_token_save_rate": output_save_rate,
            }

            if isinstance(results[task_name], dict) and len(result["results"].keys()) > 1:
                # Multiple subtasks - add to each subtask
                for subtask_name in results[task_name].keys():
                    results[task_name][subtask_name].update(inflection_metrics)
            else:
                # Single task
                results[task_name].update(inflection_metrics)

        if curr_output_path and task_name in SAVE_PREDS_TO_FILE_TASKS:
            preds = result["samples"][task_name]
            final_preds = list()
            for i, pred in enumerate(preds):
                pred.pop("metrics")
                pred = {
                    k: v
                    for k, v in pred.items()
                    if k
                    in ["doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "exact"]
                }

                if get_inflection_metrics and task_name in GENERATIVE_TASKS:
                    # Add inflection flags
                    try:
                        pred["input_has_inflections"] = (
                            1 if metrics["input_inflection_counts"][i] > 0 else 0
                        )
                        pred["output_has_inflections"] = (
                            1 if metrics["output_inflection_counts"][i] > 0 else 0
                        )
                    except:
                        pred["input_has_inflections"] = -1
                        pred["output_has_inflections"] = -1
                final_preds.append(pred)

            with open(os.path.join(curr_output_path, "preds.json"), "w") as fp:
                json.dump(final_preds, fp, indent=4)

        # Save per-run cache for this task immediately after completion
        if use_per_run_cache and per_run_cache_dir and per_run_stage_name:
            _save_per_run_cache(
                per_run_cache_dir, per_run_stage_name, task_name, results[task_name]
            )

        # Print results for this task immediately after completion
        print(f"\n{'=' * 80}")
        print(f"Results for task: {task_name}")
        print(f"{'=' * 80}")
        if isinstance(results[task_name], dict):
            for key, value in results[task_name].items():
                if isinstance(value, dict):
                    print(f"\n  {key}:")
                    for metric_name, metric_value in value.items():
                        print(f"    {metric_name}: {metric_value}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {results[task_name]}")
        print(f"{'=' * 80}\n")

    if use_cache:
        # Save results to cache
        _save_cache(cache_key, results)

    return results


def multilingual_evaluate_model(
    model,
    tokenizer,
    language,
    limited=False,
    task_names=None,
    output_path=None,
    examples_limit=None,
    get_inflection_metrics=False,
    chat_format=False,
):
    try:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval import tasks
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError(
            "Downstream benchmark evaluation requires lm-eval. Install the posthoc optional dependencies."
        ) from exc
    """
    Evaluate model on multiple benchmarks using lm-evaluation-harness.
    """
    language_to_lang_code = {
        "spanish": "es",
        "german": "de",
        "portuguese": "pt",
    }
    lang_code = language_to_lang_code.get(language, language[:2])

    curr_task_to_limited_samples = task_to_limited_samples
    if examples_limit is not None:
        curr_task_to_limited_samples = defaultdict(lambda: examples_limit)

    task_configs = {
        f"{name}_{lang_code}": {"num_fewshot": 5} for name in ("xnli", "xquad", "global_mmlu_full")
    }

    task_manager = tasks.TaskManager()
    available_tasks = set(task_manager.all_tasks)
    filtered_task_configs = {}
    for task_name in task_configs:
        if task_name in available_tasks:
            filtered_task_configs[task_name] = task_configs[task_name]
        else:
            print(f"Skipping task {task_name} - not found in LM Eval Harness")
    task_configs = filtered_task_configs
    if task_names is not None:
        task_configs = {name: task_configs[name] for name in task_names}
    if not task_configs:
        raise ValueError(f"No benchmark tasks available for {language}")

    model.eval()
    adapted_model = HFLM(
        model,
        tokenizer=tokenizer,
        trust_remote_code=True,
    )

    # Evaluate the model on each task
    results = {}
    for task_name, config in task_configs.items():
        # Evaluate the model on the task
        curr_output_path = os.path.join(output_path, task_name) if output_path else None
        if curr_output_path:
            os.makedirs(curr_output_path, exist_ok=True)
        selected_tasks, samples = evaluation_selection(
            task_manager, task_name, curr_task_to_limited_samples[task_name] if limited else None
        )
        if curr_output_path and samples is not None:
            with open(os.path.join(curr_output_path, "evaluation_samples.json"), "w") as f:
                json.dump(samples, f, indent=2)
        with torch.no_grad():
            result = simple_evaluate(
                model=adapted_model,
                tasks=selected_tasks,
                samples=samples,
                task_manager=task_manager,
                batch_size=1,
                num_fewshot=config.get("num_fewshot", None),
                gen_kwargs=config.get("gen_kwargs", None),
                log_samples=True if curr_output_path else False,
                apply_chat_template=chat_format,
            )

        if len(result["results"].keys()) > 1:
            results[task_name] = dict()
            for subtask_name in result["results"].keys():
                if task_name.startswith("global_mmlu") and subtask_name.startswith(
                    f"global_mmlu_full_{lang_code}_"
                ):
                    continue
                results[task_name][subtask_name] = result["results"][subtask_name]
        else:
            results[task_name] = result["results"][task_name]

        if get_inflection_metrics and any(
            [task_name.startswith(task_prefix) for task_prefix in GENERATIVE_TASKS]
        ):
            # Get metrics after task completion
            metrics = model.get_metrics_and_reset()

            # Reorder metrics to match evaluation order
            preds = result["samples"][task_name]
            eval_inputs = []
            for pred in preds:
                if "arguments" in pred and pred["arguments"]:
                    eval_inputs.append(
                        pred["arguments"][0][0]
                        if isinstance(pred["arguments"], list)
                        else str(pred["arguments"])
                    )
                elif "doc" in pred:
                    eval_inputs.append(str(pred["doc"]))
                else:
                    eval_inputs.append("")

            # Create reorder indices
            reorder_indices = []
            for eval_input in eval_inputs:
                for i, cached_input in enumerate(metrics["inputs"]):
                    if i not in reorder_indices and eval_input[-300:] == cached_input[-300:]:
                        reorder_indices.append(i)
                        break

            # Reorder all metric lists
            reordered_metrics = dict()
            for key in ["input_inflection_counts", "output_inflection_counts", "inputs", "outputs"]:
                if key in metrics:
                    metrics[key] = [metrics[key][i] for i in reorder_indices]

            # Calculate inflection statistics
            total_instances = len(metrics["input_inflection_counts"])
            input_with_inflections = sum(
                1 for count in metrics["input_inflection_counts"] if count > 0
            )
            output_with_inflections = sum(
                1 for count in metrics["output_inflection_counts"] if count > 0
            )
            both_with_inflections = sum(
                1
                for i, o in zip(
                    metrics["input_inflection_counts"], metrics["output_inflection_counts"]
                )
                if i > 0 and o > 0
            )

            # Calculate token save rates
            def calculate_save_rate(inflection_token_counts, total_tokens):
                if total_tokens == 0:
                    return 0.0
                saved_tokens = sum(
                    count * (token_count - 1)
                    for token_count, count in inflection_token_counts.items()
                )
                return saved_tokens / total_tokens

            input_save_rate = calculate_save_rate(
                metrics["input_inflection_token_counts"], metrics["input_total_tokens"]
            )
            output_save_rate = calculate_save_rate(
                metrics["output_inflection_token_counts"], metrics["output_total_tokens"]
            )

            # Add inflection metrics to results
            inflection_metrics = {
                "input_inflection_proportion": input_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "output_inflection_proportion": output_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "both_inflection_proportion": both_with_inflections / total_instances
                if total_instances > 0
                else 0.0,
                "input_token_save_rate": input_save_rate,
                "output_token_save_rate": output_save_rate,
            }

            if isinstance(results[task_name], dict) and len(result["results"].keys()) > 1:
                # Multiple subtasks - add to each subtask
                for subtask_name in results[task_name].keys():
                    results[task_name][subtask_name].update(inflection_metrics)
            else:
                # Single task
                results[task_name].update(inflection_metrics)

        if curr_output_path and any(
            [task_name.startswith(task_prefix) for task_prefix in SAVE_PREDS_TO_FILE_TASKS]
        ):
            preds = result["samples"][task_name]
            final_preds = list()
            for i, pred in enumerate(preds):
                pred.pop("metrics")
                pred = {
                    k: v
                    for k, v in pred.items()
                    if k
                    in ["doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "exact"]
                }

                if get_inflection_metrics and any(
                    [task_name.startswith(task_prefix) for task_prefix in GENERATIVE_TASKS]
                ):
                    # Add inflection flags
                    pred["input_has_inflections"] = (
                        1 if metrics["input_inflection_counts"][i] > 0 else 0
                    )
                    pred["output_has_inflections"] = (
                        1 if metrics["output_inflection_counts"][i] > 0 else 0
                    )
                final_preds.append(pred)

            with open(os.path.join(curr_output_path, "preds.json"), "w") as fp:
                json.dump(final_preds, fp, indent=4)

    return results
