#!/usr/bin/env bash
# Phenotype table, sample list, classes and one labels file per label set.
# Light (pandas over a few field files, a small regression each): fine on a
# login node.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

mkdir -p inputs/labels
"$PYTHON" "$HERE/phenotypes.py" --fields "$FIELDS" --pan-uk "$PAN_UK" \
    --fam "$GENOTYPES.fam" --out-dir inputs
for set in "${LABEL_SETS[@]}"; do
    name="${set%%=*}" spec="${set#*=}"
    "$DIETNET" adjust-phenotype --phenotypes inputs/phenotypes.tsv \
        --trait "${spec%%:*}" --covariates "${spec#*:}" --out "inputs/labels/$name.tsv"
done
