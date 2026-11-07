# UK Biobank traits: more SNPs from the imputed data (HapMap3+)

Draft for Matt, 2026-11-03. Follows `2026-11-03-ukbb-traits-toward-0.5.md` (A1 done:
LDpred2-auto on the 416k array SNPs gives height r 0.372, BMI 0.277).

## Why

Every model so far sees only the array's 416,084 SNPs (MAF >= 1%). Published UK
Biobank height scores near r ~ 0.5-0.6 (LDpred2, SBayesR, lasso) use ~1.1-1.4M
imputed HapMap3(+) variants, which tag far more of the common causal variation. Our
joint linear model already beats clumping/thresholding (+0.02 height, +0.11 BMI), so
the ceiling is now the SNP set more than the method. More SNPs raises the baseline
and gives the Diet Network the same inputs a strong PRS uses.

## What exists on Narval (surveyed 2026-11-03, read-only)

- **No UK Biobank WGS or exome genotypes** anywhere the lab can read (all four
  `*-hussinju` projects, `shared/UKBiobank_data` in full). Dube's project
  (group `ctb-dubemar1`, holds the raw 20168 bgen) is not readable by us.
- **UKB v3 imputed (HRC + UK10K), application 20168 (ours: same IDs as the array),
  already hard-called and lifted to GRCh38** by grenier2:
  `ctb-hussinju/shared/UKBiobank_data/20168_project_Dube/genotypes_imputation/
  plink_format_INFO0.8_GP0.9/liftover_GRCh38/per_chr_clean_version/
  ukb_v3.woWithrawalSamples.liftOverCommon.GRCh38.chr{1..22}.{bed,bim,fam}`
  (811 GB; 41.3M variants incl. rare; 486,748 samples; calls at GP >= 0.9, MaCH
  R^2 >= 0.8; ids `chr:pos38:ref:alt`). **424,142 of our 424,768 cohort samples are
  in it** (626 are not, presumably later withdrawals).
- UKB `mfi` INFO files (`shared/UKBiobank_data/Imputation_INFO_scores/`).
- GATK bundle HapMap 3.3 sites, GRCh38
  (`def-hussinju/shared/References/GRCh38/Resource-bundle_GATK/hapmap_3.3.hg38.vcf.gz`):
  3.6M polymorphic SNPs over all HapMap populations, with rsids. Not the curated EUR
  HapMap3 set; useful to map rsids to GRCh38.
- **GIANT 2014 height GWAS (Wood et al., 253k Europeans, no UK Biobank)**:
  `ctb-hussinju/cam27/DIETNET_EXP/HEIGHT/DATA/GENOTYPES_2014GIANTGWAS_SNPS/`
  (2.55M HapMap2 SNPs by rsid; 2.54M of them map through the HapMap VCF).
- Application 49731 data (Camille's) and a TOPMed-imputed 19,552-sample DietNet
  subset: wrong application / too few samples for us.

## Decisions for Matt

1. **SNP set: HapMap3+ (recommended).** bigsnpr's `map_hm3_plus.rds` (Privé et al.
   2022, 1,444,196 variants chosen to tag common variation in UKB Europeans, with
   GRCh38 positions; the standard LDpred2 set). Figshare refuses non-browser
   downloads from the cluster, so it needs downloading by hand
   (https://figshare.com/ndownloader/files/37802721) and copying to
   `$SCRATCH/dietnet_traits/resources/map_hm3_plus.rds` on Narval. Alternatives:
   plain HapMap3 (`map.rds`, 1,054,330; same issue), or the GATK HapMap sites after
   filters (~2M, non-standard, heavier).
2. **"SNPs known to be associated with height": only from GWAS without UK
   Biobank.** GIANT 2018 and 2022 include UK Biobank, so our test samples helped find
   their hits: using them to pick SNPs or as features leaks. GIANT 2014 (above) has
   no UKB samples and is safe, as features (C below) or as a prior.
3. GPU time for the Diet Network on the new SNPs (step C): ~20 A100 jobs, each
   ~3-4x the array's (~1.5 h -> ~5 h): ask before submitting.

## Work

### A. Cohort on HapMap3+ (CPU)

- `imputed/variants.R`: HM3+ map -> `chr pos_hg38 a0 a1` -> matched to the imputed
  bims by position and alleles (either orientation, no strand flips needed: same
  reference) -> per-chromosome id lists.
- `imputed/extract.sbatch` (array over chromosomes): plink2 `--bfile <imputed chr>
  --keep inputs/keep.txt --extract <ids> --maf 0.01 --geno 0.1` (hard calls at GP 0.9
  leave missing genotypes) `--make-bed`; then one merge to `exp_hm3/cohort.{bed,bim,fam}`
  with samples in `exp_maf01/cohort.fam` order (`--indiv-sort f`). Expect ~1.2-1.4M
  SNPs, ~150 GB packed.
- Report: SNPs kept per chromosome, missingness, overlap with the array SNPs.

### B. Linear benchmarks on HapMap3+ (CPU)

- `exp_hm3/bench/`: the same fold lists as `exp_maf01/bench/` (train/valid/test/ld,
  `pheno.tsv`), minus the 626 absent samples (scores merge on IID, so the array
  results can be re-scored on exactly the same 424,142 for a paired comparison).
- `GWAS=plink2 submit.sh bench-ldpred2` with `EXP_DIR=exp_hm3`, 5 folds (~2-3 h a
  fold, more memory: 128G). C+T too if cheap.
- Expect height r ~ 0.5, BMI ~ 0.3. If height stays near 0.37, the SNP count is not
  the limit and the LD reference (10k) is the next suspect.

**Done 2026-11-03.** A: `submit.sh imputed` (jobs 8424876-8424881, 8427659):
1,440,938 of 1,443,190 HM3+ variants are in the imputed data; after MAF >= 1% and
<= 10% missing, `exp_hm3/cohort` has 424,142 samples x 1,290,149 SNPs (137 GB).
B: `GWAS=plink2 bench-ldpred2` (array 8427660, ~1.7 h a fold at 128G), mean test r
over 5 folds: **height 0.396** (0.394-0.399), **BMI 0.307** (0.302-0.312), vs 0.372 /
0.277 on the array SNPs. So more SNPs adds ~+0.025-0.03 to both; BMI is now at the top
of the literature range, height still well short of ~0.5. The fit, not the SNP set,
looks like height's limit: LDSC h2 0.59 (intercept 1.63) but kept chains estimate h2
0.15-0.19, the MLE sampler still diverges, 2-23 of 30 chains pass. Over-shrinkage of
that kind points at the 10k in-sample LD reference vs a 255k GWAS. Cheap next checks:
a larger LD reference (30-50k training samples) for fold 0; LDpred2-grid / lassosum2
on the same LD (h2 fixed at the LDSC value instead of estimated); individual-level
penalised regression (bigstatsr `big_spLinReg`, no LD reference, heavier).

**lassosum2 (fold 0, 2026-11-03):** height test r **0.518** (LDpred2-auto 0.394; every
LDpred2-grid model diverged), BMI 0.289 (auto 0.302). A 40k LD reference changes
nothing (auto 0.394, lassosum2 0.518), so the limit was LDpred2's sampler on height,
not the LD reference or the SNPs. Folds 1-4: array 8467266.

**Linear baseline on HM3+, 5 folds (mean test r; model chosen on validation):**

| | height | BMI |
|---|---|---|
| lassosum2 | **0.517** (0.518 0.520 0.518 0.511 0.516) | 0.296 (0.289 0.302 0.299 0.294 0.295) |
| LDpred2-auto | 0.396 | **0.308** (0.302 0.312 0.312 0.303 0.308) |

The Diet Network on HM3+ is judged against the better of the two per trait: height
0.517 (lassosum2), BMI 0.308 (LDpred2-auto).

### Efficiency for 1.29M SNPs (agreed with Matt, 2026-11-03)

- **Streamed input statistics.** Upstream's compute_input_features_statistics.py
  holds a fold's training genotypes in memory (~330 GB at 254k x 1.29M). The same
  means and SDs follow from per-SNP genotype counts, gathered a block of rows at a
  time: `compute-input-stats --engine dietnet` (means equal to upstream's, SDs to
  1e-12; `tests/unit/test_input_stats.py`), one 48G job for all folds.
- **SNP sampling (`train --snp-sample q`, engine `dietnet`).** Each training batch
  uses a random share q of the SNPs, the same for the whole batch, scaled by n/k,
  in place of per-sample input dropout: input dropout with one mask per batch, so
  the auxiliary network and the fat layer skip the dropped SNPs (~q of the work).
  q = 1 - input_dropout keeps the rate. Check before using it at 1.29M: the array
  grid (`exp_maf01`, `dec`, b138_d50, `--bn-recalibrate 20000`, folds 0-4) with
  `--snp-sample 0.5`, paired per fold against height 0.336 / BMI 0.221.
  **Result (array 8470059):** height 0.3359 (sd 0.005) vs 0.3356; per fold -0.001
  +0.003 +0.001 +0.002 -0.003. BMI 0.2221 (sd 0.012) vs 0.2211; per fold +0.009
  +0.001 +0.010 -0.013 -0.001. No change in accuracy. Training pass 19.8 -> 13.6 s an
  epoch (genotype reading and decoding are not reduced), ~10% more epochs to the
  best model, so ~25% less GPU time a model. Used for the HM3+ grid.
- **Chunked reads of samples and SNPs (`train --chunk-rows 2048 --chunks-per-buffer 4
  --snp-blocks 64 --snp-sample 0.5`; Matt's choice over 220 GB jobs, 2026-11-04).**
  Random-row batches need the whole 137 GB packed copy in the page cache: at 160G the
  pilot ran at 54 s an epoch until the cache was evicted, then over an hour; 220G
  bills ~1.7 GPUs. Instead the sorted training rows are cut into contiguous chunks,
  shuffled per epoch; each buffer reads 4 chunks in file order and shuffles them in
  memory (~59 batches), and keeps half of 64 contiguous SNP blocks (`ChunkPlan`). Not
  yet checked for accuracy: whole genome regions are dropped together, and a batch
  comes from 4 stretches of the file. Before using it at 1.29M, the same check as
  `--snp-sample`: array grid (`exp_maf01`, `dec`, b138_d50, `--bn-recalibrate 20000`,
  folds 0-4), paired against height 0.336 / BMI 0.221, plus r(row, label) to confirm
  the file's order says nothing about the traits.
  **Result (array 8543719, 2026-11-05):** the file's order is unrelated to the traits
  (r(row, label) -0.0007 height, -0.0013 BMI; chunk-mean variance 0.98 / 0.93 of
  chance). Height 0.3333 (sd 0.005) vs 0.3356 plain: per fold -0.003 +0.002 -0.002
  -0.005 -0.003. BMI 0.2210 (sd 0.009) vs 0.2211: per fold +0.004 -0.002 +0.008 -0.010
  -0.001. At most ~0.002 lower; peak memory 19-23 GB a job on 416k SNPs (~60-70 GB
  expected at 1.29M), no page cache needed; ~30-40 min a model. Used for the HM3+
  grid (array 8553904, 128G); the unchunked 220G pilot was cancelled (validation and
  BN recalibration read ~105k scattered rows an epoch, ~12 min an epoch).
- **AlphaGenome for the new SNPs** (for `dec_ag` on HM3+): 1,155,048 HM3+ SNPs not
  already scored, on the Mac (`~/dietnet_alphagenome/ukbb_hm3/run_scoring.sh`,
  resumable, ~2.5 days at the API's rate); then merged with the array table.

### Results on HM3+ (2026-11-05)

Chunked reads (`--chunk-rows 2048 --chunks-per-buffer 4 --snp-blocks 64
--snp-sample 0.5`), `--bn-recalibrate 20000`, b138_d50, trained in place on Lustre
(`STAGE=0`; packed rows read with file reads through gaps, evaluation rows cached in
memory: ~70-100 s an epoch, ~155 s with free weights). Mean held-out r over 5 folds:

| | height | BMI |
|---|---|---|
| Diet Network `dec` | 0.394 | 0.260 |
| Diet Network `gwas` (A2: per-fold GWAS z, beta, signed -log10 p) | 0.392 | 0.230 |
| LDpred2-auto | 0.396 | **0.308** |
| lassosum2 | 0.517 | 0.296 |

A2: GWAS features do no better than decile frequencies (height) or worse (BMI); runs
peak early (best epoch 3 on 3 of 5 height folds).

**Penalized Diet Network (B3).** The model called "hybrid" until 2026-11-06, in full
"prior-informed penalized Diet Network": `W_j = aux(emb_j) + delta_j` with a group-lasso
penalty on `delta`. It fits per-SNP weights jointly on individual-level genotypes, like a
penalized regression PRS (lasso / snpnet, or lassosum2 from summary statistics), and
shrinks them toward what the auxiliary network predicts from the SNP's features, not
toward zero, like an annotation-informed PRS (SBayesRC, LDpred-funct). It uses no GWAS
summary statistics, though `dec` features are frequencies per trait decile on the fold's
training samples, so they carry marginal SNP-trait signal. It is not a linear score: each
SNP has 100 weights feeding hidden layers. In code it stays `hybrid.py`,
`--free-snp-weights` and the `fw<lambda>` run directories. `penDN-dec` in tables.

**Lambda grid on fold 0 (raw R^2, not rescaled):**

| lambda | height r | height R^2 | BMI r | BMI R^2 | SNPs with a free weight (height) |
|---|---|---|---|---|---|
| 0.03 | 0.442 | 0.195 | 0.183 | 0.030 | |
| 0.1 | 0.490 | 0.232 | 0.237 | 0.054 | |
| **0.3** | **0.543** | **0.290** | **0.320** | **0.102** | ~1.29M (all) |
| 1 | 0.445 | 0.198 | 0.289 | 0.083 | ~0.9M |
| 3 | 0.411 | 0.168 | 0.279 | 0.076 | ~0.6M |
| 6 | 0.398 | 0.157 | 0.271 | 0.073 | ~7k |
| plain `dec` | 0.393 | 0.154 | 0.257 | 0.066 | - |

Only lambda 0.3 beats the best linear model on fold 0; below it the best epoch comes
earlier (height epoch 9 at 0.1 and 8 at 0.03; BMI 7 and 3; 62-102 at 0.3 for height), which
would fit overfitting. The grid places the optimum between 0.1 and 1 without pinning it.
Lambda was compared on fold 0's test r, and the plan records that choosing on validation
agrees. Folds 1-4 played no part: they average 0.539 (height) against fold 0's 0.543.

**Final comparison, 5 folds (2026-11-06), every model evaluated the same way:** R^2 and
RMSE after fitting the label on the score on validation (`benchmark/rescale_eval.py`,
using `score_eval.evaluate`), test r unchanged by that. Mean r / R^2:

| | height r | height R^2 | BMI r | BMI R^2 |
|---|---|---|---|---|
| **`penDN-dec`, lambda 0.3** | **0.540** | **0.292** | **0.318** | **0.101** |
| lassosum2 | 0.517 | 0.267 | 0.296 | 0.087 |
| LDpred2-auto | 0.396 | 0.157 | 0.308 | 0.095 |
| Diet Network `dec` | 0.394 | 0.155 | 0.260 | 0.068 |
| Diet Network `gwas` | 0.392 | 0.154 | 0.230 | 0.053 |

Per fold 0-4 (test r; R^2):

| | r | R^2 |
|---|---|---|
| Height, `penDN-dec` | 0.5433 0.5420 0.5384 0.5360 0.5404 | 0.2951 0.2938 0.2899 0.2873 0.2921 |
| Height, lassosum2 | 0.5178 0.5202 0.5176 0.5114 0.5156 | 0.2680 0.2706 0.2678 0.2615 0.2659 |
| Height, LDpred2-auto | 0.3936 0.3976 0.3988 0.3959 0.3932 | 0.1549 0.1581 0.1590 0.1568 0.1546 |
| Height, `dec` | 0.3925 0.3992 0.3901 0.3970 0.3916 | 0.1540 0.1594 0.1521 0.1576 0.1533 |
| Height, `gwas` | 0.3932 0.3957 0.3904 0.3927 0.3881 | 0.1545 0.1566 0.1524 0.1542 0.1506 |
| BMI, `penDN-dec` | 0.3196 0.3179 0.3219 0.3068 0.3216 | 0.1021 0.1011 0.1036 0.0940 0.1034 |
| BMI, LDpred2-auto | 0.3022 0.3119 0.3123 0.3034 0.3079 | 0.0913 0.0973 0.0975 0.0920 0.0947 |
| BMI, lassosum2 | 0.2885 0.3022 0.2987 0.2940 0.2948 | 0.0832 0.0913 0.0891 0.0864 0.0868 |
| BMI, `dec` | 0.2569 0.2682 0.2671 0.2558 0.2520 | 0.0660 0.0719 0.0713 0.0653 0.0635 |
| BMI, `gwas` | 0.2194 0.2281 0.2409 0.2363 0.2274 | 0.0482 0.0520 0.0580 0.0557 0.0517 |

`penDN-dec` against the best PRS for each trait, paired by fold: height vs lassosum2, R^2
+0.025 (+0.027 +0.023 +0.022 +0.026 +0.026), r +0.024; BMI vs LDpred2-auto, R^2 +0.006
(+0.011 +0.004 +0.006 +0.002 +0.009), r +0.010. Higher on every fold for both traits. The
folds share training samples, so these are not independent and carry no p-value.

Rescaling mattered only for some models. Raw R^2 (`dietnet check`) to rescaled: `penDN-dec`
height 0.287 to 0.292, BMI 0.1006 to 0.1008; `gwas` BMI 0.043 to 0.053; `gwas` height 0.153
to 0.154; `dec` 0.155 and 0.067 to 0.155 and 0.068.

Sources on Narval (`exp_hm3`): `rescale/<label>_<spec>_<config>.json` (per-fold metrics and
the match with training, at most 8.5e-7 from the saved predictions), `.npz` / `.tsv`
(whole-cohort predictions; stay on the cluster), `bench/fold<k>/<label>.ldpred2.plink.json`
and `.multi.json` (LDpred2-auto, lassosum2), `logs/check_*.out` (the grid). C+T was not run
on HM3+: 0.351 / 0.165 are from the array SNPs and are not comparable.

Evaluating needed the matched cohort that `dietnet predict` builds (547 GB, one byte per
genotype, 3 h on Lustre; deleted afterwards), then ~35-47 min per run of 5 models. That run
found `memmap_inputs` was copy-on-write, which the kernel refuses for a dataset larger than
RAM; now read-only (commit 4f71386). The job's 128 GB of "used" memory includes reclaimable
page cache from the map, so it does not measure what `predict` needs.

**AlphaGenome for the 1,155,048 new SNPs: scored 2026-11-06.** 3,853 columns, as the array
table. On Narval: `exp_hm3/alphagenome/alphagenome_hm3.tsv.gz` (20 GB; SHA-256 checked against
the Mac). Not yet merged with the array table. Backup on the Mac,
`~/dietnet_alphagenome/ukbb_hm3/`: the table, its parts and the array table
(`alphagenome_ukbb_array.tsv.gz`). `dec_ag` mixes in UKB-derived `dec` features, so it stays
on Narval. `run_scoring.sh`'s watchdog kills an attempt that writes no part for 15 min,
which includes the final combine (~1.5 h): stop the script once all parts exist, or raise
`STALL`, or it retries forever.

**Next, agreed with Matt: AlphaGenome as a prior.** Plain `dec_ag` paired by fold against
`dec`, and `penDN-dec_ag` at lambda 0.3 against `penDN-dec`, 5 folds, both traits, evaluated
as above.

### C. Diet Network on HapMap3+ (GPU, after B and Matt's go)

- Dataset, partition from the fold lists (same folds, by sample id), per-fold
  stats (~3.4x memory: large-memory nodes), `dec` and `dec_ag` embeddings
  (AlphaGenome scores exist only for array SNPs: score the new SNPs first, ~2 days of
  API, or restrict `dec_ag` to the overlap), then `dec`, plus `gwas` / `gwasld` /
  hybrid variants from the first spec as they land.
- New, leakage-free external prior: a `giant2014` feature block (GIANT 2014 z, BETA,
  -log10 P per SNP, mapped rsid -> GRCh38 via the HapMap VCF; fold-independent
  `precomputed`), variants `dec_giant` and `dec_giant_ag`.

## Not doing

- WGS: not available to us. If Dube's project holds the 500k WGS, that is a data
  request to make outside this work.
- Rare variants (MAF < 1%): they broke the Diet Network's standardisation before.
