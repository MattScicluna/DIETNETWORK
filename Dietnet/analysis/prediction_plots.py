"""
Utilities for visualizing ensemble prediction breakdowns.

Provides a stacked bar plot for a target population where each sample's bar
shows the vote counts across source populations from the compact prediction text.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_compact_predictions(path: Path) -> Dict[str, List[Tuple[str, int]]]:
    """
    Parse compact prediction text:
        <sample_id> CEUGBR(15) ACB(1)

    Returns a mapping: sample_id -> list of (label, count)
    """
    results: Dict[str, List[Tuple[str, int]]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sample_id = parts[0]
        label_counts: List[Tuple[str, int]] = []
        for token in parts[1:]:
            token = token.rstrip(".,;")
            if "(" in token and token.endswith(")"):
                label, count_str = token[:-1].split("(", 1)
                try:
                    count = int(count_str)
                except ValueError:
                    count = 1
            else:
                label = token
                count = 1
            label_counts.append((label, count))
        results[sample_id] = label_counts
    return results


def plot_population_stack(
    predictions_path: str,
    labels_path: str,
    target_population: str,
    cmap_path: str,
    output_path: str,
    title: str = None,
) -> None:
    """
    Build a stacked bar plot for a target population using compact predictions.

    Each bar corresponds to one sample in the target population.
    Heights are vote counts; colors come from the provided colormap.
    Bars are sorted left-to-right by the maximum vote fraction (descending).
    """
    pred_map = _parse_compact_predictions(Path(predictions_path))
    labels_df = pd.read_csv(labels_path, sep="\t")
    label_id_col = labels_df.columns[0]
    label_class_col = labels_df.columns[1]
    labels_df[label_id_col] = labels_df[label_id_col].astype(str)
    labels_df[label_class_col] = labels_df[label_class_col].astype(str)

    with open(cmap_path, "r") as f:
        cmap = json.load(f)

    # Filter to target population samples present in predictions
    target_df = labels_df[
        (labels_df[label_class_col] == target_population)
        & (labels_df[label_id_col].isin(pred_map.keys()))
    ].copy()

    if target_df.empty:
        raise ValueError(f"No samples for population '{target_population}' found in predictions.")

    # Build per-sample counts and compute max fraction for sorting
    samples_info = []
    for _, row in target_df.iterrows():
        sid = row[label_id_col]
        counts = pred_map.get(sid, [])
        total_votes = sum(c for _, c in counts) if counts else 0
        if total_votes == 0:
            continue
        max_fraction = max(c for _, c in counts) / total_votes
        samples_info.append((sid, counts, total_votes, max_fraction))

    if not samples_info:
        raise ValueError(f"No vote data available for population '{target_population}'.")

    # Sort by max vote fraction descending to create a gradient
    samples_info.sort(key=lambda x: -x[3])

    # Determine all labels present (order by overall total descending)
    label_totals: Dict[str, int] = {}
    for _, counts, _, _ in samples_info:
        for lbl, cnt in counts:
            label_totals[lbl] = label_totals.get(lbl, 0) + cnt
    ordered_labels = [lbl for lbl, _ in sorted(label_totals.items(), key=lambda item: -item[1])]

    # Build stacked bars
    x_positions = np.arange(len(samples_info))
    bottoms = np.zeros(len(samples_info))

    plt.figure(figsize=(max(8, len(samples_info) * 0.02), 4))

    for lbl in ordered_labels:
        heights = []
        for _, counts, _, _ in samples_info:
            height = next((cnt for l, cnt in counts if l == lbl), 0)
            heights.append(height)
        heights = np.array(heights)
        if not heights.any():
            continue
        plt.bar(
            x_positions,
            heights,
            bottom=bottoms,
            color=cmap.get(lbl, "#888888"),
            edgecolor="none",
            width=1.0,
            label=lbl,
        )
        bottoms += heights

    plt.xlim(0, len(samples_info))
    plt.ylim(0, max(bottoms) if bottoms.size else 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title or f"{target_population} prediction breakdown", fontsize=9)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, ncol=2)
    plt.tight_layout()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
