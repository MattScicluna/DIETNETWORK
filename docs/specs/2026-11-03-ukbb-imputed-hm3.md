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
