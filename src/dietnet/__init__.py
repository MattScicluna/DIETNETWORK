"""Diet Network for genetic ancestry prediction.

The model, training loop and the statistics it is trained on are HussinLab's
published code (Rochefort-Boulanger, Scicluna et al., AJHG 2026), shipped
unmodified and called from here; see :mod:`dietnet.upstream`. This package adds
what the published scripts leave to the user: building their HDF5 input from
PLINK, allele-aware matching of new cohorts, portable model packages, ensembles,
biobank-scale inference and a command line around all of it.
"""

__version__ = "0.3.0"
