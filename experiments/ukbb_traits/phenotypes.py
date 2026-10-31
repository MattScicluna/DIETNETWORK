"""The phenotype table, sample list and class labels for the UK Biobank trait runs.

Reads one CSV per UK Biobank field (``variable_id,sample_id,instance,index,value``)
and the Pan-UKB ancestry table with its bridge to this application's ids, keeps
the first assessment visit (instance 0), and writes to ``--out-dir``:

- ``phenotypes.tsv``: sample, height (50), bmi (21001), sex (31), age (21022),
  townsend (22189), and Pan-UKB's pop and related, for the kept samples;
- ``keep.txt``: FID and IID of the kept samples, for ``plink2 --keep``: in the genotype
  ``.fam``, Pan-UKB ``EUR``, and no field above missing, so that every trait
  and covariate set is trained on the same samples with the same folds;
- ``classes.tsv``: one class for everyone. Partitioning needs class labels; a
  regression run on pooled frequencies uses them for nothing else.

Relatedness is not applied here: the samples stay in the dataset, and a
partition can leave out Pan-UKB's ``related`` ones later without rebuilding it.

    python phenotypes.py --fields DIR --pan-uk DIR --fam GENOTYPES.fam --out-dir DIR
"""

import argparse
from pathlib import Path

import pandas as pd

FIELDS = {"height": 50, "bmi": 21001, "sex": 31, "age": 21022, "townsend": 22189}
PAN_UKB = "all_pops_non_eur_pruned_within_pop_pc_covs.tsv"
BRIDGE = "ukb20168bridge31063.txt"


def field(directory: Path, number: int) -> pd.Series:
    table = pd.read_csv(directory / f"{number}.csv.gz")
    first = table[(table["instance"] == 0) & (table["index"] == 0)]
    if first["sample_id"].duplicated().any():
        raise ValueError(f"field {number}: several instance-0 values for a sample")
    return pd.to_numeric(first.set_index("sample_id")["value"], errors="raise")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fields", type=Path, required=True, help="one CSV.gz per field")
    parser.add_argument("--pan-uk", type=Path, required=True, help="Pan-UKB table and bridge")
    parser.add_argument("--fam", type=Path, required=True, help="the genotypes' .fam")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    table = pd.DataFrame({name: field(args.fields, n) for name, n in FIELDS.items()})
    bridge = pd.read_csv(args.pan_uk / BRIDGE, sep=r"\s+", header=None, names=["sample", "s"])
    pan = pd.read_csv(args.pan_uk / PAN_UKB, sep="\t", usecols=["s", "pop", "related"])
    pan = bridge.merge(pan, on="s").set_index("sample")[["pop", "related"]]
    table = table.join(pan, how="inner")

    fam = pd.read_csv(args.fam, sep=r"\s+", header=None, usecols=[0, 1], names=["fid", "iid"])
    genotyped = set(fam["iid"][fam["iid"] > 0])  # withdrawn samples are negative ids
    counts = {"with genotypes and Pan-UKB ancestry": table.index.isin(genotyped).sum()}
    table = table[table.index.isin(genotyped)]
    table = table[table["pop"] == "EUR"]
    counts["EUR"] = len(table)
    table = table.dropna(subset=list(FIELDS))
    counts["EUR, no field missing"] = len(table)
    counts["of which Pan-UKB related"] = int(table["related"].sum())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.index.name = "sample"
    table.sort_index().to_csv(args.out_dir / "phenotypes.tsv", sep="\t")
    # FID and IID as in the .fam: plink2 reads a one-column --keep file as FIDs
    # and keeps no one.
    keep = fam[fam["iid"].isin(table.index)].sort_values("iid")
    keep.to_csv(args.out_dir / "keep.txt", sep="\t", index=False, header=False)
    classes = pd.DataFrame({"sample": sorted(table.index), "class": "all"})
    classes.to_csv(args.out_dir / "classes.tsv", sep="\t", index=False)
    for what, n in counts.items():
        print(f"{n:>8,}  {what}")


if __name__ == "__main__":
    main()
