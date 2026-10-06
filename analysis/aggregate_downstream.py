"""Normalize saved downstream_metrics.json files into a tidy CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _walk_metrics(value: Any, prefix: tuple[str, ...] = ()):
    """Yield numeric metric leaves from the task/method nesting in eval JSON."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "alias":
                continue
            yield from _walk_metrics(child, prefix + (str(key),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if prefix:
            yield prefix, float(value)


def flatten_results(payload: dict[str, Any], *, model: str, language: str) -> pd.DataFrame:
    """Flatten task -> method -> metric JSON (including nested task variants).

    For nested evaluator output, the task path is retained as ``task`` joined
    with ``/``; metric names remain exactly as saved by the evaluator.
    """
    rows = []
    for task, methods in payload.items():
        if not isinstance(methods, dict):
            continue
        for method, result in methods.items():
            if method == "alias":
                continue
            for path, metric_value in _walk_metrics(result):
                if len(path) == 1:
                    metric = path[0]
                    task_name = str(task)
                else:
                    task_name = "/".join((str(task), *path[:-1]))
                    metric = path[-1]
                rows.append(
                    {
                        "model": model,
                        "language": language,
                        "task": task_name,
                        "method": str(method),
                        "metric": metric,
                        "value": metric_value,
                    }
                )
    return pd.DataFrame(rows, columns=["model", "language", "task", "method", "metric", "value"])


def run(manifest_path: Path, output_path: Path) -> None:
    manifest = pd.read_csv(manifest_path)
    required = {"model", "language", "path"}
    if required - set(manifest.columns):
        raise ValueError("manifest must have model,language,path columns")
    tables = []
    for row in manifest.to_dict("records"):
        source = Path(row["path"]).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        with source.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object in {row['path']}")
        tables.append(
            flatten_results(payload, model=str(row["model"]), language=str(row["language"]))
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=["model", "language", "task", "method", "metric", "value"])
    )
    result.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with model,language,path")
    parser.add_argument("--output", type=Path, required=True, help="destination tidy CSV")
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
