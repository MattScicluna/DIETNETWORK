"""The published models (Rochefort-Boulanger, Scicluna et al., AJHG 2026) as packages.

Zenodo record 21133813 (concept doi:10.5281/zenodo.16943452), file
``Dietnet_training.tar.gz``, holds the 15 trained models (3 replicates x 5
folds), the configs they were trained with (one per replicate; they differ
only in seed), the per-fold SNP embeddings and normalisation statistics, and
the 1KGP training dataset. :func:`import_training_release` turns that into
model packages, so ``dietnet predict`` runs the published models.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from dietnet import package

ZENODO_RECORD = "21133813"
CONCEPT_DOI = "10.5281/zenodo.16943452"
TRAINING_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD}/files/Dietnet_training.tar.gz?download=1"
)
N_REPLICATES, N_FOLDS = 3, 5


def fetch(dest) -> Path:
    """Download and unpack Dietnet_training.tar.gz into ``dest``; return its directory."""
    import json

    dest = Path(dest)
    release = dest / "Dietnet_training"
    if (release / "supporting_files").is_dir():
        return release
    dest.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}") as r:
        record = json.load(r)
    (entry,) = [f for f in record["files"] if f["key"] == "Dietnet_training.tar.gz"]
    algorithm, expected = entry["checksum"].split(":", 1)

    archive = dest / "Dietnet_training.tar.gz"
    urllib.request.urlretrieve(TRAINING_URL, archive)
    digest = hashlib.new(algorithm, archive.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{archive}: {algorithm} {digest} != {expected} from Zenodo")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    return release


def import_training_release(release, out) -> list[package.Package]:
    """Packages for the 15 published models from an unpacked Dietnet_training/."""
    release, out = Path(release), Path(out)
    support = release / "supporting_files"
    embeddings = np.load(support / "SNPs_embedding.npz")["emb"]
    stats = np.load(support / "normalization_stats.npz")

    packages = []
    for rep in range(1, N_REPLICATES + 1):
        config = yaml.safe_load((support / f"config_rep{rep}models.yaml").read_text())
        for fold in range(N_FOLDS):
            packages.append(
                package.build(
                    out / f"seed_{config['seed']}" / f"fold_{fold}",
                    checkpoint=release / "model_weights" / f"model_rep{rep}_fold{fold}.pt",
                    config=config,
                    embedding=embeddings[fold],
                    means=stats["means_by_fold"][fold],
                    sds=stats["sd_by_fold"][fold],
                    train_dataset=release / "1KGP_dataset.hdf5",
                    task="classification",
                    normalize=True,
                    seed=config["seed"],
                    fold=fold,
                    extra={
                        "source": f"Zenodo record {ZENODO_RECORD} (doi:{CONCEPT_DOI})",
                        "replicate": rep,
                    },
                )
            )
    return packages
