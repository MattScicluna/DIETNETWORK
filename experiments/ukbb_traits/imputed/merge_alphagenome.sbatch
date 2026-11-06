#!/usr/bin/env bash
# The AlphaGenome table for the HM3+ cohort: the array table at the cohort's positions
# (its other SNPs would only cost memory when the table is read whole), then the new SNPs.
# Submit from the experiment directory with ARRAY_TABLE and NEW_TABLE in the environment.
#SBATCH --job-name=traits_ag_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
set -euo pipefail
: "${ARRAY_TABLE:?}" "${NEW_TABLE:?}"
out=alphagenome/alphagenome_merged.tsv.gz
{
    zcat "$ARRAY_TABLE" \
        | awk -F'\t' 'NR==FNR {keep["chr"$1":"$4]=1; next} FNR==1 || (($1":"$2) in keep)' \
            cohort.bim -
    zcat "$NEW_TABLE" | tail -n +2
} | gzip -1 > "$out.partial"
gzip -t "$out.partial"
rows=$(zcat "$out.partial" | tail -n +2 | wc -l)
expected=$(zcat "$NEW_TABLE" | head -1 | awk -F'\t' '{print NF}' || true)
columns=$(zcat "$out.partial" | head -1 | awk -F'\t' '{print NF}' || true)
echo "rows $rows, columns $columns (expected $expected), cohort SNPs $(wc -l < cohort.bim)"
[ "$columns" = "$expected" ] || { echo "unexpected column count"; exit 1; }
[ "$rows" -ge "$(wc -l < cohort.bim)" ] || { echo "fewer rows than cohort SNPs"; exit 1; }
mv "$out.partial" "$out"
