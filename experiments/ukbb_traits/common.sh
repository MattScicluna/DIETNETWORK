# Sourced by every step. Paths come from the environment (see README.md), never
# from the repository: keep them in an env file on the cluster and `source` it.
set -euo pipefail
: "${GENOTYPES:?plink prefix of the genotypes}" "${FIELDS:?directory of <field>.csv.gz}"
: "${PAN_UK:?directory of the Pan-UKB table and bridge}" "${ALPHAGENOME:?AlphaGenome table}"
: "${DIETNET_VENV:?}" "${EXP_DIR:?}" "${HERE:?the directory of these scripts}"
DIETNET="$DIETNET_VENV/bin/dietnet"
PYTHON="$DIETNET_VENV/bin/python"

# Label sets, <name>=<trait>:<covariates>. Each gets its own dataset (the labels
# live in it), all over the same samples in the same order, so they share one
# partition, one set of statistics and the embeddings.
LABELS="${LABELS:-height=height:sex,age bmi=bmi:sex,age height_tdi=height:sex,age,townsend bmi_tdi=bmi:sex,age,townsend}"
read -ra LABEL_SETS <<< "$LABELS"
FIRST="${LABEL_SETS[0]%%=*}"  # the dataset partition, stats and embeddings are read from
SPECS="${SPECS:-freq ag freq_ag freq_random freq_shuffled}"
read -ra SPEC_NAMES <<< "$SPECS"
NB_FOLDS=5

mkdir -p "$EXP_DIR/logs"
cd "$EXP_DIR"
