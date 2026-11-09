#!/usr/bin/env bash
# Phenotype table, sample list, classes and one labels file per label set,
# with the label's deciles as classes for the decile features (specs/dec*.yaml).
# Light (pandas over a few field files, a small regression each): fine on a
# login node.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

mkdir -p inputs/labels inputs/deciles
"$PYTHON" "$HERE/phenotypes.py" --fields "$FIELDS" --pan-uk "$PAN_UK" \
    --fam "$GENOTYPES.fam" --out-dir inputs
for set in "${LABEL_SETS[@]}"; do
    name="${set%%=*}" spec="${set#*=}"
    "$DIETNET" adjust-phenotype --phenotypes inputs/phenotypes.tsv \
        --trait "${spec%%:*}" --covariates "${spec#*:}" --out "inputs/labels/$name.tsv"
    # Cut points over the whole cohort; the per-fold frequencies use training
    # samples only, so test labels enter only through the nine cut points.
    "$PYTHON" -c "
import sys, pandas as pd
labels = pd.read_csv(sys.argv[1], sep='\t', index_col=0).iloc[:, 0]
deciles = pd.qcut(labels, 10, labels=[f'd{i:02d}' for i in range(1, 11)])
deciles.rename('decile').to_csv(sys.argv[2], sep='\t')
" "inputs/labels/$name.tsv" "inputs/deciles/$name.tsv"
done
