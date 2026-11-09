"""A single polygenic score, evaluated as ct_select.py evaluates its chosen threshold.

Reads plink2's score files ``<prefix>.valid.sscore`` and ``<prefix>.test.sscore``,
fits the label on the score on validation (for R^2 and RMSE on the label's scale)
and writes the test r, R^2 and RMSE to a JSON file, together with the key/value
pairs of ``<prefix>.info.tsv`` if it exists.

    python score_eval.py --pheno bench/pheno.tsv --label height \
        --prefix bench/fold0/height.ldpred2 --method "regenie + LDpred2-auto" \
        --out bench/fold0/height.ldpred2.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def scores(path, label, pheno):
    s = pd.read_csv(path, sep="\t")
    s = s.rename(columns={"#FID": "FID", "#IID": "IID"}).astype({"IID": str})
    # The score itself; plink2 also writes NAMED_ALLELE_DOSAGE_SUM, an allele count.
    merged = s[["IID", "SCORE1_SUM"]].merge(pheno[["IID", label]], on="IID")
    return merged["SCORE1_SUM"].to_numpy(float), merged[label].to_numpy(float)


def evaluate(xv, yv, xt, yt):
    slope, intercept = np.polyfit(xv, yv, 1)
    pred = intercept + slope * xt
    return {
        "valid_r": float(np.corrcoef(xv, yv)[0, 1]),
        "test_n": int(len(yt)),
        "test_r": float(np.corrcoef(xt, yt)[0, 1]),
        "test_r2": float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
        "test_rmse": float(np.sqrt(((yt - pred) ** 2).mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pheno = pd.read_csv(args.pheno, sep="\t", dtype={"IID": str})

    xv, yv = scores(f"{args.prefix}.valid.sscore", args.label, pheno)
    xt, yt = scores(f"{args.prefix}.test.sscore", args.label, pheno)
    result = {"method": args.method, "label": args.label, **evaluate(xv, yv, xt, yt)}
    info = f"{args.prefix}.info.tsv"
    if os.path.exists(info):
        result["info"] = dict(pd.read_csv(info, sep="\t").itertuples(index=False))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("label", "valid_r", "test_r", "test_r2")}))


if __name__ == "__main__":
    main()
