# UK Biobank traits: toward height r ~ 0.5 with the Diet Network + AlphaGenome

Spec agreed with Matt on 2026-11-03. Branch `feat/ukbb-traits` (PR #19), worktree
`.claude/worktrees/ukbb-traits`. Read the plan's "Resume here" section first.

## Where things stand

All on the MAF-filtered run directory `$SCRATCH/dietnet_traits/exp_maf01` on Narval:
424,768 Pan-UKB EUR samples, 416,084 SNPs (MAF >= 1%), labels height and BMI
(adjusted for sex and age, standardised), 5 folds (same partition everywhere).

Mean held-out r over the 5 folds (same test samples throughout):

| | height | BMI |
|---|---|---|
| Diet Network, decile features (`dec`) | 0.336 | 0.221 |
| Diet Network, deciles + AlphaGenome (`dec_ag`) | 0.334 | 0.219 |
| regenie GWAS + clumping/thresholding (`submit.sh bench`) | 0.351 | 0.165 |
| naive marginal score (fold 0, `linear_check.py`) | 0.33 | 0.18 |
| target: joint linear models (literature, UKB-only) | ~0.5-0.6 | ~0.25-0.3 |

Diet Network settings: `configs/b138_d50.yaml` (input dropout 0.5),
`TRAIN_FLAGS="--bn-recalibrate 20000"`, `RUN_SUFFIX=_bn`. AlphaGenome adds nothing so
far (per-fold differences of mixed sign).

**Why the Diet Network sits at the marginal score.** Each SNP's fat-layer weight is
`aux(features of that SNP)`. With per-SNP marginal features the model can only learn
"weight = f(own association)": a marginal score, with every SNP in an LD block
counted again. C+T removes part of that redundancy (0.35); joint models remove it
(~0.5). The auxiliary network sees each SNP alone, so it cannot adjust a SNP's weight
for its neighbours, and AlphaGenome (what a variant does) says nothing about LD.

## Work, in order

### A1. Benchmark: regenie -> LDpred2-auto (CPU)

The per-fold regenie GWAS already exists (`bench/fold<k>/gwas_<label>.regenie`,
training samples only; columns CHROM GENPOS ID ALLELE0 ALLELE1 A1FREQ N TEST BETA SE
CHISQ LOG10P; BETA is per ALLELE1). Replace C+T with LDpred2-auto (bigsnpr 1.12.21,
installed for `module load StdEnv/2023 gcc r/4.4.0`, library
`~/R/x86_64-pc-linux-gnu-library/4.4`):

- LD from the fold's 10,000-sample LD reference (`bench/fold<k>/ld.txt`) on the
  array SNPs (`snp_readBed2` on `cohort.bed` with those rows; `snp_cor` within
  3 cM or 1 Mb; genetic positions via `snp_asGeneticPos` if maps are available,
  else a physical window). In-sample LD avoids the HapMap3/array SNP mismatch of
  published panels.
- `snp_ldpred2_auto` (several chains, `allow_jump_sign = FALSE`, `shrink_corr = 0.95`),
  keep converged chains, average betas.
- Score valid and test with plink2 `--score` (column `SCORE1_SUM`, not
  `NAMED_ALLELE_DOSAGE_SUM`: that bug already cost one run), report test r, R^2,
  RMSE per fold to `bench/fold<k>/<label>.ldpred2.json`.
- New files: `benchmark/ldpred2.R`, `benchmark/ldpred2.sbatch` (array over folds),
  a `submit.sh bench-ldpred2` phase. Test first on one chromosome (`CHR=22`).
- Expect height r ~ 0.45-0.55. If far below, check allele alignment first.

**Result (2026-11-03, arrays 8282169 + 8299197, `GWAS=plink2`): height r = 0.372
(sd 0.003), BMI r = 0.277 (sd 0.005), mean test r over 5 folds** (per fold:
height 0.369 0.375 0.375 0.374 0.368; BMI 0.274 0.283 0.280 0.273 0.274), ~47 min
a fold on 32 CPUs, 55 GB.

- **regenie's GWAS does not suit LDpred2.** Step 2 associates y minus the LOCO
  polygenic prediction, not y, so chi^2 is inflated (fold 0 LDSC h2 0.92,
  intercept 1.27) and LDpred2-auto on it gave height 0.179, BMI 0.130 (fold 0).
  `GWAS=plink2` runs a plain marginal GWAS (plink2 `--glm`, training samples only,
  `bench/fold<k>/gwas_marginal_<label>.regenie`, regenie's columns), which gives
  the results above. A2/B2 should take their GWAS features from these files too.
- Allele alignment checked: bigsnpr's `allele1` is the bim's first allele, matched
  by ID with no flips; scores use the bim's A1; LD itself does not depend on the
  counted allele.
- The MLE update of p and h2 (bigsnpr's default) diverges in every chain, every
  fold, both traits; the script falls back to `use_MLE = FALSE`, where 3-11 of 30
  chains pass the tutorial's convergence filter. Kept chains estimate h2 ~0.14
  (height) and ~0.05 (BMI): too low, a sign of LD-reference mismatch at N ~250k.
- A 3 Mb window (fold 0, `LD_WINDOW_KB=3000`) changes r by < 0.003 (height 0.370,
  BMI 0.276), though chain h2 rises to 0.36 / 0.10: the window is not the limit.
- So the joint model gains +0.02 over C+T for height, +0.11 for BMI; BMI is in the
  literature's range, height short of 0.45-0.55. Likely reasons, untested: 416k
  array SNPs vs ~1.1M imputed HapMap3 SNPs in the literature (less tagging of
  causal variants), and a 10k LD reference for a 255k GWAS. Possible next checks:
  a larger LD reference (e.g. 30-50k training samples), or individual-level
  penalised regression (bigstatsr `big_spLinReg`), which needs no LD reference.

### Shared code: per-fold precomputed features

**Done 2026-11-03** (`src/dietnet/features.py`, tests in `tests/unit/test_features.py`,
`docs/features.md`). Per-fold tables must also share feature columns and cover the
same SNPs (otherwise the `missing` column would be all zeros in some fold).


Needed by A2 and B2. `build-embedding`'s `precomputed` provider is fold-independent
(SNPs x features). Add a per-fold form: `path:` containing `{fold}` (e.g.
`features/gwas_height_fold{fold}.tsv`) is read once per fold and gives a
folds x SNPs x features block, matched and oriented exactly as now (`orient: effect`
negates where the dataset counts REF). Tests: a per-fold table equals stacking the
per-fold builds; missing fold file is an error naming it; mixing with fold-dependent
`genotype_freq` blocks works. Document in `docs/features.md`.

### A2. Diet Network with regenie features (third variant)

Per fold, a table from that fold's regenie GWAS: chrom, pos, ref = ALLELE0,
alt = ALLELE1, features z = BETA/SE, BETA, -log10 P (signed by BETA). Variants
`gwas`, `gwas_ag` (+ AlphaGenome block as in `dec_ag`). Specs as templates
(`{label}`, filled by `embedding.sbatch`), fold file names via `{fold}`.
20 GPU jobs (height, BMI x 2 variants x 5 folds), b138_d50 + recalibration.
This tests "does AlphaGenome help on top of a strong per-SNP signal?".

### B2 (Matt's option 2). LD-aware SNP features

Give the auxiliary network what it needs to discount LD redundancy. Per fold,
from the training GWAS and the fold's 10k LD reference:

- LD score: sum of r^2 with SNPs within 1 Mb (plink2 `--r2-unphased` on `ld.txt`,
  `--ld-window-kb 1000`, `--ld-window-r2 0.01`, `--ld-window 99999`).
- Neighbourhood statistics within 1 Mb: max |z|, z_j / max|z| (how far below the
  local lead), LD-weighted neighbour z (sum_k r_jk z_k / sum_k r_jk^2), and whether
  j is a clump index (from the C+T clumps, which are per fold already).
- Variants `gwasld` (z, BETA, signed -log10 P + LD features), `gwasld_ag`.
- `benchmark/ld_features.py` (or `features/ld_features.py` under the experiment)
  writes the per-fold tables; a CPU array job over folds. 20 GPU jobs.
- Expected: some gain over `gwas` (marginal), short of joint models.

### B3 (Matt's option 3). Hybrid: free per-SNP weights on top of the auxiliary network

`W_j = aux(emb_j) + delta_j`, with `delta` (hidden x SNPs, initialised 0) trained
with a group-lasso penalty `lambda * sum_j ||delta_j||_2` (one group per SNP, so it
selects SNPs) applied as a proximal step after each optimiser step. This contains
penalised regression (lasso-like, the ~0.5-0.6 regime) as a special case, with the
SNP features (deciles, GWAS, AlphaGenome) acting as a prior, as annotations do in
SBayesRC. Where AlphaGenome has its best chance to help.

Implementation (never edit `Dietnet_train_and_evaluate/`):

- In `src/dietnet/` (engine `dietnet` only): a wrapper that adds `delta` to the
  upstream model and changes the fat-layer weights passed to `main_net` from
  `aux_out.T` to `aux_out.T + delta`, without changing upstream files (subclass or
  wrap `DietNetwork.forward` at runtime, as `_launch.py` does for other hooks).
- Its own optimiser (Adam, `lr_delta`) and the proximal group-lasso step; CLI
  options `--free-snp-weights LAMBDA` (0 = off, the default, unchanged behaviour)
  and `--free-snp-lr`.
- Memory: 416k x 100 floats = 166 MB, plus optimiser state ~0.5 GB: fine on A100.
- Checkpoints and packages carry `delta`; `package.py` loads it strictly and
  `predict` applies it (bump the package format version; old packages still load).
- Tests: delta = 0 / lambda -> infinity equals the plain model bit for bit on the
  synthetic fixture; the proximal step matches its closed form; a package round
  trip predicts what training reported for its test fold; integration run on
  synthetic data reaches the plain model's quality.
- **Implemented 2026-11-05** (`src/dietnet/hybrid.py`, `train --free-snp-weights`,
  `--free-snp-lr`; tests `tests/unit/test_hybrid.py` and the hybrid tests in
  `tests/integration/test_parallel.py`: a penalty keeping delta at 0 trains the plain
  model bit for bit; a format-3 package predicts what training reported). **Lambda's
  scale:** delta is stepped by Adam, which moves each entry by ~lr whatever the
  gradient, so a SNP's column moves ~lr * sqrt(hidden) = 10 lr a step against a
  proximal shrinkage of lr * lambda: lambda ~ 10 zeroes everything and the values
  below (1e-6 ... 1e-4) are no penalty at all. Grid instead: 0.3, 1, 3, 6.
- Runs: lambda grid on fold 0 (e.g. 1e-6, 1e-5, 1e-4, chosen on validation), then
  `hybrid_dec`, `hybrid_dec_ag` (and `hybrid_gwas_ag` if A2 helps) on 5 folds.
- Success: height r >= ~0.5 over 5 folds; AlphaGenome judged by paired per-fold
  differences.

### A3 (optional). SBayesRC with vs without AlphaGenome annotations

The linear counterpart of `dec_ag` vs `dec`. Needs GCTB SBayesRC, its LD reference
(HapMap3-based, ~1.1M SNPs) and annotations for those SNPs. AlphaGenome scores exist
only for the 485,516 array SNPs, so either restrict to the overlap or score the
HapMap3 SNPs (~1.1M, about 2 days of API). Decide after A1/B3.

## Operational notes (all learned the hard way)

- Narval needs MFA: if `ssh -o BatchMode=yes narval` asks for it, Matt runs
  `ssh -fN narval` and approves; the control socket is under `~/.ssh/`.
- Nothing heavy on the login node, including `dietnet check` (it loads torch):
  summaries go through `sbatch --wait` CPU jobs.
- GitHub over port 22 is flaky from Matt's network; push with
  `GIT_SSH_COMMAND='ssh -o Hostname=ssh.github.com -o Port=443 -o HostKeyAlias=github.com'`.
- Commit with `PATH=.venv/bin:$PATH` so the hooks run; black may reformat and abort
  the commit: re-add and commit again.
- The session's sandbox refuses complex inline shell: put watchers in script files
  under `$CLAUDE_JOB_DIR/tmp` and run those in the background.
- Modules: regenie needs `StdEnv/2020 gcc/9.3.0 regenie/3.2.1`; plink2 needs
  `StdEnv/2023 plink/2.0.0-a.6.32`.
- Run recipe (Diet Network):

  ```bash
  cd $SCRATCH/dietnet_traits && source traits.env
  export EXP_DIR=$SCRATCH/dietnet_traits/exp_maf01 LABELS="height=height:sex,age bmi=bmi:sex,age"
  TRAIN_LABELS="height bmi" TRAIN_SPECS="dec dec_ag" TRAIN_CONFIGS=b138_d50 SEEDS=78 \
  FOLDS="0 1 2 3 4" TRAIN_FLAGS="--bn-recalibrate 20000" RUN_SUFFIX=_bn \
  bash repo/experiments/ukbb_traits/submit.sh train
  ```

  New feature sets: `SPECS="<spec>@height <spec>@bmi"` with `embedding.sbatch`
  (see the `dec@...` submission in the plan) before training.
