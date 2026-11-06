#!/usr/bin/env bash
# Whole-cohort predictions of the given runs, evaluated as the polygenic scores are
# (rescale_eval.py: rescaled on validation). Submit from the experiment directory, with
# BENCH (the directory of rescale_eval.py and score_eval.py) and RUNS ("<spec>/<config>"
# under runs/<label>/) in the environment. The matched cohort (547 GB at 1.29M SNPs) is
# built once, reused by every run, and removed if everything succeeds.
#SBATCH --job-name=traits_rescale
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -uo pipefail
: "${BENCH:?}" "${RUNS:?}" "${DIETNET_VENV:?}"
mkdir -p rescale/work
status=0
for label in height bmi; do
    for run in $RUNS; do
        tag="${label}_${run//\//_}"
        echo "== $tag $(date '+%F %T')"
        "$DIETNET_VENV/bin/dietnet" predict --model "runs/$label/$run/packages" --plink cohort \
            --work-dir rescale/work --out "rescale/$tag.tsv" --scores "rescale/$tag.npz" \
            && PYTHONPATH="$BENCH" "$DIETNET_VENV/bin/python" "$BENCH/rescale_eval.py" \
                --scores "rescale/$tag.npz" --packages "runs/$label/$run/packages" \
                --partition "partition_fold{fold}.npz" --pheno bench/pheno.tsv --label "$label" \
                --out "rescale/$tag.json" \
            || { echo "FAILED: $tag"; status=1; }
    done
done
[ "$status" = 0 ] && rm -rf rescale/work
exit "$status"
