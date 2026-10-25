#!/usr/bin/env bash
# Genotype I/O benchmark (benchmarks/bench_genotype_io.py) as a batch job.
#
# Paths come from the environment so that no cluster path is committed:
#
#   PLINK_PREFIX=/path/to/cohort OUT_JSON=/path/to/results.json \
#   DIETNET_VENV=/path/to/.venv [WORKDIR=/path/on/lustre] [FORMATS=...] \
#   sbatch --account=ACCOUNT benchmarks/slurm/bench_io.sbatch
#
# WORKDIR defaults to the node-local $SLURM_TMPDIR. Point it at scratch to time
# the shared filesystem instead. Memory: about 2.2x the int8 matrix
# (samples x SNPs bytes); the defaults below fit UK Biobank (~487k x 170k).
#SBATCH --job-name=dietnet_bench_io
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out

set -euo pipefail

: "${PLINK_PREFIX:?set PLINK_PREFIX}" "${OUT_JSON:?set OUT_JSON}" "${DIETNET_VENV:?set DIETNET_VENV}"
WORKDIR="${WORKDIR:-${SLURM_TMPDIR:?not in a SLURM job}/bench}"
FORMATS="${FORMATS:-bed-pyplink,bed-numpy,npy,hdf5-contiguous,hdf5-chunked-rows}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# sbatch copies the script, so BASH_SOURCE points at the copy; prefer the submit dir.
[ -f "$REPO/benchmarks/bench_genotype_io.py" ] || REPO="$SLURM_SUBMIT_DIR"

echo "node $(hostname)  workdir $WORKDIR  formats $FORMATS"
df -h "$(dirname "$WORKDIR")" | tail -1

"$DIETNET_VENV/bin/python" -u "$REPO/benchmarks/bench_genotype_io.py" \
    --plink-prefix "$PLINK_PREFIX" \
    --workdir "$WORKDIR" \
    --repeats 1 \
    --formats "$FORMATS" \
    --out "$OUT_JSON"

# Leave nothing behind on shared storage.
case "$WORKDIR" in "$SLURM_TMPDIR"*) ;; *) rm -rf "$WORKDIR" ;; esac
