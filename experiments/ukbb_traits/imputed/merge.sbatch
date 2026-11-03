#!/usr/bin/env bash
# The 22 chromosomes of extract.sbatch as one fileset, EXP_DIR/cohort.{bed,bim,fam},
# and the linear benchmarks' sample lists from SRC_EXP/bench restricted to its samples.
# Every chromosome has the same samples in the same order (checked), so the .bed
# files concatenate: one 3-byte header, then each file's variant-major blocks.
#SBATCH --job-name=traits_imputed_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
source "$HERE/common.sh"
: "${SRC_EXP:?}"

for c in $(seq 2 22); do
    cmp -s imputed/chr1.fam "imputed/chr$c.fam" || { echo "chr$c.fam differs" >&2; exit 1; }
done
cp imputed/chr1.fam cohort.fam
for c in $(seq 1 22); do cat "imputed/chr$c.bim"; done > cohort.bim
{
    head -c 3 imputed/chr1.bed
    for c in $(seq 1 22); do tail -c +4 "imputed/chr$c.bed"; done
} > cohort.bed
# Size check: header + variants x ceil(samples / 4) bytes.
n=$(wc -l < cohort.fam) m=$(wc -l < cohort.bim)
[ "$(stat -c %s cohort.bed)" -eq $((3 + m * ((n + 3) / 4))) ] || { echo "bad bed size" >&2; exit 1; }
echo "cohort: $n samples, $m variants"

# Benchmark lists: the source's folds, samples absent here dropped.
mkdir -p bench
awk 'NR == FNR { keep[$2]; next } FNR == 1 || ($2 in keep)' cohort.fam "$SRC_EXP/bench/pheno.tsv" \
    > bench/pheno.tsv
for d in "$SRC_EXP"/bench/fold*; do
    k=$(basename "$d")
    mkdir -p "bench/$k"
    for s in train valid test ld; do
        awk 'NR == FNR { keep[$2]; next } $2 in keep' cohort.fam "$d/$s.txt" > "bench/$k/$s.txt"
    done
    wc -l "bench/$k"/{train,valid,test,ld}.txt | head -4 | tr '\n' ' '; echo
done
