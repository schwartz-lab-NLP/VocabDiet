"""Aggregate saved Patchscopes resolution rows into paper-facing summaries.

Input manifests are CSV files with ``model,language,path`` columns. Each path
points to a ``patchscopes_resolution_layer.csv`` file. Raw experiment outputs
are deliberately not bundled with this repository.
"""

from __future__ import annotations

import argparse
from urllib.parse import quote
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"word", "type", "word_n_tokens", "is_single_type", "first_layer_additive"}


def category_for_type(value: str) -> str:
    """Map the paper's transformation labels to their coarse classes."""
    if value == "add_capitalization":
        return "capitalization"
    if ";" in value:
        return "inflection"
    if (
        value.startswith("-")
        or value.endswith("-")
        or value in {"non-", "pre-", "re-", "anti-", "un-"}
    ):
        return "derivation"
    return "other"


def _rate(frame: pd.DataFrame, success: pd.Series) -> float:
    return float(success.mean()) if len(frame) else float("nan")


def summarize_resolution(
    frame: pd.DataFrame, *, single_type_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-transformation and class-weighted IV/OOV success tables.

    ``embed_success`` means the additive representation first matches at layer
    zero. ``detok_success`` means it matches at layers zero through ten. IV/OOV are
    defined by the target's original token count (one vs. more than one).
    """
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"resolution CSV is missing columns: {', '.join(sorted(missing))}")
    data = frame.copy()
    if single_type_only:
        data = data[data["is_single_type"].astype(str).str.lower().isin({"true", "1"})]
    data["word_n_tokens"] = pd.to_numeric(data["word_n_tokens"], errors="coerce")
    layer = pd.to_numeric(data["first_layer_additive"], errors="coerce")
    data["embed_success"] = layer.eq(0)
    data["detok_success"] = layer.between(0, 10)
    data["category"] = data["type"].astype(str).map(category_for_type)
    data = data[data["category"].isin({"inflection", "derivation", "capitalization"})]

    rows = []
    for (transformation, category), group in data.groupby(["type", "category"], sort=True):
        record = {"type": transformation, "category": category}
        for split, mask in (
            ("iv", group["word_n_tokens"].eq(1)),
            ("oov", group["word_n_tokens"].gt(1)),
        ):
            subset = group[mask]
            record[f"{split}_n"] = int(len(subset))
            record[f"{split}_embed_success"] = _rate(subset, subset["embed_success"])
            record[f"{split}_detok_success"] = _rate(subset, subset["detok_success"])
        rows.append(record)
    per_type = pd.DataFrame(
        rows,
        columns=[
            "type",
            "category",
            "iv_n",
            "iv_embed_success",
            "iv_detok_success",
            "oov_n",
            "oov_embed_success",
            "oov_detok_success",
        ],
    )

    class_rows = []
    for category, group in data.groupby("category", sort=True):
        record = {"category": category}
        for split, mask in (
            ("iv", group["word_n_tokens"].eq(1)),
            ("oov", group["word_n_tokens"].gt(1)),
        ):
            subset = group[mask]
            record[f"{split}_n"] = int(len(subset))
            record[f"{split}_embed_success"] = _rate(subset, subset["embed_success"])
            record[f"{split}_detok_success"] = _rate(subset, subset["detok_success"])
        class_rows.append(record)
    return per_type, pd.DataFrame(class_rows)


def exemplar_correlations(per_type: pd.DataFrame) -> pd.DataFrame:
    """Compute exploratory Spearman tests for IV exemplar count vs success.

    This uses log(IV target count), matching the saved failure-analysis
    convention. It is a transparent recomputation from raw rows, not a claim
    that the paper's reported transformation-selection recipe is recovered.
    """
    from scipy.stats import spearmanr

    rows = []
    for category, group in [("all", per_type), *per_type.groupby("category", sort=True)]:
        valid_iv = group[group["iv_n"].gt(0) & group["iv_embed_success"].notna()]
        valid_oov = valid_iv[valid_iv["oov_n"].gt(0) & valid_iv["oov_embed_success"].notna()]
        for split, data, success_col in (
            ("iv", valid_iv, "iv_embed_success"),
            ("oov", valid_oov, "oov_embed_success"),
        ):
            rho = pvalue = float("nan")
            if len(data) >= 3:
                result = spearmanr(np.log(data["iv_n"]), data[success_col])
                rho, pvalue = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "category": category,
                    "target_split": split,
                    "n_transformations": len(data),
                    "spearman_rho": rho,
                    "pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    expected = {"model", "language", "path"}
    if expected - set(manifest.columns):
        raise ValueError("manifest must have model,language,path columns")
    return manifest


def run(
    manifest_path: Path, output_dir: Path, *, all_types: bool = False, figure: bool = False
) -> None:
    manifest = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest.to_dict("records"):
        source = Path(entry["path"]).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        raw = pd.read_csv(source)
        per_type, by_class = summarize_resolution(raw, single_type_only=not all_types)
        correlations = exemplar_correlations(per_type)
        prefix = quote(str(entry["model"]), safe="") + "_" + quote(str(entry["language"]), safe="")
        per_type.insert(0, "language", entry["language"])
        per_type.insert(0, "model", entry["model"])
        by_class.insert(0, "language", entry["language"])
        by_class.insert(0, "model", entry["model"])
        per_type.to_csv(output_dir / f"{prefix}_patchscopes_by_type.csv", index=False)
        by_class.to_csv(output_dir / f"{prefix}_patchscopes_by_class.csv", index=False)
        correlations.insert(0, "language", entry["language"])
        correlations.insert(0, "model", entry["model"])
        correlations.to_csv(output_dir / f"{prefix}_exemplar_correlations.csv", index=False)
        if figure and not by_class.empty:
            _plot_class_summary(by_class, output_dir / f"{prefix}_iv_oov_success.png")


def _plot_class_summary(summary: pd.DataFrame, path: Path) -> None:
    """Save the paper-relevant IV/OOV additive-success comparison."""
    import matplotlib.pyplot as plt

    categories = summary["category"].tolist()
    x = np.arange(len(categories))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - width / 2, summary["iv_embed_success"] * 100, width, label="IV (one token)")
    ax.bar(x + width / 2, summary["oov_embed_success"] * 100, width, label="OOV (multiple tokens)")
    ax.set_xticks(x, categories)
    ax.set_ylabel("Additive embedding success (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--all-types", action="store_true", help="include rows not marked is_single_type"
    )
    parser.add_argument(
        "--figure", action="store_true", help="also save an IV/OOV additive-success plot"
    )
    args = parser.parse_args()
    run(args.manifest, args.output_dir, all_types=args.all_types, figure=args.figure)


if __name__ == "__main__":
    main()
