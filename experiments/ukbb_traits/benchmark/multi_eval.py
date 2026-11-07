"""Many polygenic scores at once: per model family, the model chosen on validation.

Reads plink2's ``<prefix>.valid.sscore`` and ``<prefix>.test.sscore``, scored from
``<prefix>.betas.tsv`` with one column per model (names like ``grid_...`` and
``lasso_...``; plink2 calls them SCORE1_SUM, ... in that order),
picks per family (the name's first ``_`` part) the model whose score correlates
best with the label on validation, and evaluates it on test as score_eval.py does.

    python multi_eval.py --pheno bench/pheno.tsv --label height \
        --prefix bench/fold0/height.ldpred2.plink.multi --out bench/fold0/height.multi.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from score_eval import evaluate


def read(path, label, pheno, names):
    """plink2 names the scores SCORE1_SUM, SCORE2_SUM, ... in the betas file's order."""
    s = pd.read_csv(path, sep="\t").rename(columns={"#FID": "FID", "#IID": "IID"})
    s = s.rename(columns={f"SCORE{i + 1}_SUM": f"{n}_SUM" for i, n in enumerate(names)})
    s = s.astype({"IID": str}).merge(pheno[["IID", label]], on="IID")
    return s, [f"{n}_SUM" for n in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pheno", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pheno = pd.read_csv(args.pheno, sep="\t", dtype={"IID": str})
    names = pd.read_csv(f"{args.prefix}.betas.tsv", sep="\t", nrows=0).columns[2:].tolist()
    valid, cols = read(f"{args.prefix}.valid.sscore", args.label, pheno, names)
    test, _ = read(f"{args.prefix}.test.sscore", args.label, pheno, names)

    yv, yt = valid[args.label].to_numpy(float), test[args.label].to_numpy(float)
    valid_r = {
        c[: -len("_SUM")]: float(np.corrcoef(valid[c], yv)[0, 1]) if valid[c].std() > 0 else np.nan
        for c in cols
    }
    result = {"label": args.label, "families": {}}
    for family in sorted({name.split("_")[0] for name in valid_r}):
        mine = {n: r for n, r in valid_r.items() if n.startswith(family + "_") and np.isfinite(r)}
        best = max(mine, key=mine.get)
        xv, xt = valid[best + "_SUM"].to_numpy(float), test[best + "_SUM"].to_numpy(float)
        result["families"][family] = {
            "model": best,
            "models": len(mine),
            **evaluate(xv, yv, xt, yt),
        }
    result["valid_r"] = valid_r
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps({f: (v["model"], round(v["test_r"], 4)) for f, v in result["families"].items()})
    )


if __name__ == "__main__":
    main()
