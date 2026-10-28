"""Fake UK Biobank-shaped inputs around the kgp_mini fixture, for dryrun.sh.

python make_fake.py tests/fixtures OUT_DIR
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fixtures, out = Path(sys.argv[1]), Path(sys.argv[2])
rng = np.random.default_rng(0)
for d in ("geno", "fields", "pan"):
    (out / d).mkdir(parents=True, exist_ok=True)

shutil.copy(fixtures / "kgp_mini.bed", out / "geno/ukb.bed")
shutil.copy(fixtures / "kgp_mini.bim", out / "geno/ukb.bim")
fam = pd.read_csv(fixtures / "kgp_mini.fam", sep=" ", header=None)
ids = np.arange(1000001, 1000001 + len(fam))
fam[0] = ids
fam[1] = ids
fam.to_csv(out / "geno/ukb.fam", sep=" ", header=False, index=False)

# Genotyped samples are complete; extra, ungenotyped ones are partly missing and
# must be dropped.
extra = np.arange(2000001, 2000041)
everyone = np.concatenate([ids, extra])
values = {
    50: rng.normal(170, 9, len(everyone)),
    21001: rng.normal(27, 4, len(everyone)),
    31: rng.integers(0, 2, len(everyone)).astype(float),
    21022: rng.integers(40, 70, len(everyone)).astype(float),
    22189: rng.normal(-1, 3, len(everyone)),
}
for field, v in values.items():
    rows = pd.DataFrame(
        {"variable_id": field, "sample_id": everyone, "instance": 0, "index": 0, "value": v}
    )
    rows = rows[~(np.isin(rows["sample_id"], extra) & (rng.random(len(rows)) < 0.5))]
    later = rows.head(30).assign(instance=1, value=-999.0)  # a later visit: ignored
    pd.concat([rows, later]).to_csv(out / f"fields/{field}.csv.gz", index=False)

pan_ids = everyone + 5_000_000
bridge = pd.DataFrame({"ours": everyone, "pan": pan_ids})
bridge.to_csv(out / "pan/ukb20168bridge31063.txt", sep=" ", header=False, index=False)
related = np.where(rng.random(len(pan_ids)) < 0.1, "true", "false")
pd.DataFrame({"s": pan_ids, "PC1": 0.0, "pop": "EUR", "related": related}).to_csv(
    out / "pan/all_pops_non_eur_pruned_within_pop_pc_covs.tsv", sep="\t", index=False
)

bim = pd.read_csv(fixtures / "kgp_mini.bim", sep="\t", header=None)
parts = bim[1].str.split(":", expand=True)
table = pd.DataFrame(
    {"chrom": parts[0], "pos": parts[1].astype(int), "ref": parts[2], "alt": parts[3]}
)
tracks = pd.DataFrame(rng.normal(size=(len(table), 80)), columns=[f"T|t{t}" for t in range(80)])
pd.concat([table, tracks], axis=1).to_csv(out / "alphagenome.tsv.gz", sep="\t", index=False)
