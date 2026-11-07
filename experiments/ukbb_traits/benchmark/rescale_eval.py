"""Diet Network packages evaluated as the polygenic scores are: rescaled on validation.

``score_eval.py`` fits the label on the score on validation before computing test R^2
and RMSE; ``dietnet check`` uses the model's raw predictions. Packages keep only their
test predictions, so this takes per-model scores for the whole cohort (``dietnet
predict --scores``), evaluates each model on its own fold with ``score_eval.evaluate``,
and checks that the test predictions match the ones saved at training.

    python rescale_eval.py --scores rescale/height_fw0.3.npz \
        --packages runs/height/dec/b138_d50_bn_chunk_fw0.3/packages \
        --partition "partition_fold{fold}.npz" --pheno bench/pheno.tsv --label height \
        --out rescale/height_fw0.3.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from score_eval import evaluate

# Saved predictions are float32; anything larger than this is a different model output.
MATCH_TOLERANCE = 1e-3


def fold_of(model_name: str) -> tuple[int, int]:
    """Seed and fold from a package name such as ``seed78_fold3``."""
    seed, fold = model_name.split("_fold")
    return int(seed.removeprefix("seed")), int(fold)


def evaluate_model(scores, samples, labels, valid_ids, test_ids, saved) -> dict:
    """One model: raw and validation-rescaled test metrics, and the match with training."""
    index = pd.Series(np.arange(len(samples)), index=samples)
    xv, xt = scores[index.loc[valid_ids].to_numpy()], scores[index.loc[test_ids].to_numpy()]
    yv, yt = labels.loc[valid_ids].to_numpy(float), labels.loc[test_ids].to_numpy(float)
    if np.isnan(yv).any() or np.isnan(yt).any():
        raise ValueError("a validation or test sample has no label")

    saved = saved.set_index("sample").loc[test_ids]
    return {
        **evaluate(xv, yv, xt, yt),
        "raw_test_r2": float(1 - ((yt - xt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
        "valid_n": int(len(yv)),
        "max_abs_diff_saved_prediction": float(np.abs(xt - saved["prediction"]).max()),
        "max_abs_diff_saved_label": float(np.abs(yt - saved["label"]).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--packages", required=True, type=Path)
    parser.add_argument("--partition", required=True, help="Path with {fold}.")
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    data = np.load(args.scores, allow_pickle=False)
    samples = data["samples"].astype(str)
    pheno = pd.read_csv(args.pheno, sep="\t", dtype={"IID": str}).set_index("IID")

    folds = []
    for i, name in enumerate(data["models"].astype(str)):
        seed, fold = fold_of(name)
        part = np.load(args.partition.format(fold=fold), allow_pickle=True)["folds_samples"][0]
        valid_ids, test_ids = part[1].astype(str), part[2].astype(str)
        saved = pd.read_csv(
            args.packages / f"seed_{seed}" / f"fold_{fold}" / "test_predictions.tsv",
            sep="\t",
            dtype={"sample": str},
        )
        row = evaluate_model(
            data["scores"][i], samples, pheno[args.label], valid_ids, test_ids, saved
        )
        folds.append({"seed": seed, "fold": fold, **row})

    table = pd.DataFrame(folds)
    means = table[["test_r", "test_r2", "raw_test_r2", "test_rmse"]].mean().to_dict()
    with open(args.out, "w") as f:
        json.dump({"label": args.label, "folds": folds, "mean": means}, f, indent=2)

    columns = ["fold", "valid_r", "test_r", "test_r2", "raw_test_r2", "test_rmse"]
    print(table[columns].round(4).to_string(index=False))
    print("mean", json.dumps({k: round(v, 4) for k, v in means.items()}))
    worst = table[["max_abs_diff_saved_prediction", "max_abs_diff_saved_label"]].max()
    print(f"largest difference from training: {worst.to_dict()}")
    if (worst > MATCH_TOLERANCE).any():
        raise SystemExit(f"predictions differ from training by more than {MATCH_TOLERANCE}")


if __name__ == "__main__":
    main()
