"""Clumping and thresholding: the p-value threshold chosen on validation, scored on test.

Reads plink2's per-threshold score files ``<prefix>.<split>.<threshold>.sscore``
for the validation and test samples, picks the threshold whose score correlates
best with the label on validation, and writes that threshold's test r, R^2 (of a
linear fit of the label on the score, fitted on validation) and the r of every
threshold to a JSON file.

    python ct_select.py --pheno bench/pheno.tsv --label height \
        --prefix bench/fold0/height --thresholds 5e-8 ... 1 --out bench/fold0/height.json
"""

import argparse
import json

import numpy as np
import pandas as pd


def scores(prefix, split, threshold, label, pheno):
    s = pd.read_csv(f"{prefix}.{split}.{threshold}.sscore", sep="\t")
    s = s.rename(columns={"#FID": "FID", "#IID": "IID"}).astype({"IID": str})
    # The score itself; plink2 also writes NAMED_ALLELE_DOSAGE_SUM, an allele count.
    merged = s[["IID", "SCORE1_SUM"]].merge(pheno[["IID", label]], on="IID")
    return merged["SCORE1_SUM"].to_numpy(float), merged[label].to_numpy(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--thresholds", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pheno = pd.read_csv(args.pheno, sep="\t", dtype={"IID": str})

    valid_r = {}
    for t in args.thresholds:
        x, y = scores(args.prefix, "valid", t, args.label, pheno)
        valid_r[t] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else float("nan")
    best = max((t for t in valid_r if np.isfinite(valid_r[t])), key=lambda t: valid_r[t])

    xv, yv = scores(args.prefix, "valid", best, args.label, pheno)
    slope, intercept = np.polyfit(xv, yv, 1)  # the score on the label's scale, from validation
    xt, yt = scores(args.prefix, "test", best, args.label, pheno)
    pred = intercept + slope * xt
    result = {
        "method": "regenie + clumping and thresholding",
        "label": args.label,
        "threshold": float(best),
        "valid_r": valid_r,
        "test_n": int(len(yt)),
        "test_r": float(np.corrcoef(xt, yt)[0, 1]),
        "test_r2": float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
        "test_rmse": float(np.sqrt(((yt - pred) ** 2).mean())),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("label", "threshold", "test_r", "test_r2")}))


if __name__ == "__main__":
    main()
