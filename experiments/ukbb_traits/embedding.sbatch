#!/usr/bin/env bash
# One auxiliary-network input per feature spec (array task i: the i-th of
# SPECS, from specs/<name>.yaml). The AlphaGenome table (485k SNPs x 3,849
# tracks) is read whole and reduced by PCA, hence the memory.
#SBATCH --job-name=traits_embedding
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=08:00:00
source "$HERE/common.sh"

name="${SPEC_NAMES[$SLURM_ARRAY_TASK_ID]}"
ln -sfn "$ALPHAGENOME" alphagenome.tsv.gz  # the specs' path, relative to EXP_DIR
mkdir -p embeddings
"$DIETNET" build-embedding --dataset "data/$FIRST.h5" --partition partition.npz \
    --spec "$HERE/specs/$name.yaml" --out "embeddings/$name.npz"
