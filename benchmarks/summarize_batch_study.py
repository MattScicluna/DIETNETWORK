"""Held-out accuracy per batch-size setting of benchmarks/slurm/batch_size_study.sbatch.

    python benchmarks/summarize_batch_study.py OUT

prints one row per (setting, fold) and a mean per setting, from each model
package's test_predictions.tsv, with the epoch the best model came from.
"""

import json
import sys
from pathlib import Path

import pandas as pd

rows = []
for setting in sorted(Path(sys.argv[1]).glob("batch*_lr*x")):
    for pred in sorted(setting.glob("exp/packages/seed_*/fold_*/test_predictions.tsv")):
        t = pd.read_csv(pred, sep="\t")
        meta = json.loads((pred.parent / "metadata.json").read_text())
        rows.append(
            {
                "setting": setting.name,
                "fold": int(pred.parent.name.split("_")[1]),
                "accuracy": (t.prediction == t.label).mean(),
                "results": Path(meta.get("results_dir", "")).name[:0] or "",
            }
        )
df = pd.DataFrame(rows).drop(columns="results")
print(df.pivot(index="fold", columns="setting", values="accuracy").round(4).to_string())
summary = df.groupby("setting").accuracy.agg(["mean", "std", "count"]).round(4)
print(summary.to_string())
