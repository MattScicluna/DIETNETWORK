# LDpred2-auto on one fold's regenie GWAS, with LD from the fold's LD reference.
#
#   Rscript ldpred2.R <ld.bed> <fold dir> <labels, comma-separated> <tag> <ncores> <tmp dir> \
#       [<GWAS prefix, default gwas>]
#
# <ld.bed>: the fold's 10,000 LD-reference samples (training samples only), all
# SNPs of the GWAS. <fold dir> holds <prefix>_<label>.regenie (regenie's columns;
# BETA per ALLELE1). LD is computed once, on the SNPs
# with a finite BETA and SE in every label's GWAS, so all labels share it.
# Writes, per label, <fold dir>/<label>.ldpred2<tag>.betas.tsv (ID, A1, BETA:
# input to plink2 --score) and <label>.ldpred2<tag>.info.tsv (key, value).
suppressPackageStartupMessages({
  library(bigsnpr)
  library(data.table)
})
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) %in% 6:7)
gwas_prefix <- if (length(args) == 7) args[7] else "gwas"
ld_bed <- args[1]
fold_dir <- args[2]
labels <- strsplit(args[3], ",")[[1]]
tag <- args[4]
ncores <- as.integer(args[5])
tmp <- args[6]
WINDOW_KB <- 1000  # physical window: no GRCh38 genetic map offline on the nodes
N_CHAINS <- 30
bigparallelr::set_blas_ncores(1)
set.seed(1)

log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")

# The LD reference, mean-imputed (array genotypes have a few missing calls).
rds <- snp_readBed(ld_bed, backingfile = file.path(tmp, "ldref"))
obj <- snp_attach(rds)
G <- snp_fastImputeSimple(obj$genotypes, method = "mean2", ncores = ncores)
map <- obj$map
map <- data.frame(chr = as.integer(map$chromosome), rsid = map$marker.ID,
                  pos = map$physical.pos, a1 = map$allele1, a0 = map$allele2)
log_msg("LD reference:", nrow(G), "samples,", ncol(G), "SNPs")

read_gwas <- function(label) {
  g <- fread(file.path(fold_dir, paste0(gwas_prefix, "_", label, ".regenie")))
  g <- g[TEST == "ADD" & is.finite(BETA) & is.finite(SE) & SE > 0]
  data.frame(chr = as.integer(g$CHROM), pos = g$GENPOS, rsid = g$ID,
             a0 = g$ALLELE0, a1 = g$ALLELE1, beta = g$BETA, beta_se = g$SE, n_eff = g$N)
}
gwas <- lapply(setNames(labels, labels), read_gwas)
shared <- Reduce(intersect, lapply(gwas, function(g) g$rsid))

# Match by ID (same genotypes, so no strand flips); snp_match also flips BETA
# wherever a GWAS counts the other allele, so every beta is per map$a1.
matched <- lapply(gwas, function(g) {
  m <- snp_match(g[g$rsid %in% shared, ], map, join_by_pos = FALSE, strand_flip = FALSE,
                 match.min.prop = 0.9)
  m[order(m$`_NUM_ID_`), ]
})
ids <- matched[[1]]$`_NUM_ID_`
for (m in matched) stopifnot(identical(m$`_NUM_ID_`, ids))
log_msg("matched SNPs:", length(ids))

# QC (the LDpred2 tutorial's): a SNP must vary in the LD reference, and the
# genotype SD the GWAS implies (sd_y / sqrt(n se^2 + beta^2), with sd_y
# estimated as the median ratio) must agree with the LD reference's.
sd_ldref <- sqrt(big_colstats(G, ind.col = ids, ncores = ncores)$var)
bad <- !is.finite(sd_ldref) | sd_ldref < 0.05
for (label in labels) {
  m <- matched[[label]]
  sd_raw <- 1 / sqrt(m$n_eff * m$beta_se^2 + m$beta^2)
  sd_y <- median(sd_ldref[!bad] / sd_raw[!bad])
  sd_ss <- sd_y * sd_raw
  is_bad <- sd_ss < 0.5 * sd_ldref | sd_ss > sd_ldref + 0.1 | sd_ss < 0.1
  log_msg(label, ": sd_y estimate", signif(sd_y, 4), ";", sum(is_bad & !bad), "SNPs fail the sd check")
  bad <- bad | is_bad
}
log_msg("QC removes", sum(bad), "SNPs (", sum(sd_ldref < 0.05 | !is.finite(sd_ldref)),
        "barely vary in the LD reference)")
ids <- ids[!bad]
matched <- lapply(matched, function(m) m[!bad, ])

# Correlations per chromosome, in map order (the order of every matched table).
corr <- NULL
ld <- numeric(0)
for (chr in unique(map$chr[ids])) {
  cols <- ids[map$chr[ids] == chr]
  corr0 <- snp_cor(G, ind.col = cols, size = WINDOW_KB, infos.pos = map$pos[cols],
                   ncores = ncores)
  if (any(!is.finite(corr0@x))) stop("chromosome ", chr, ": non-finite LD correlations")
  ld <- c(ld, Matrix::colSums(corr0^2))
  if (is.null(corr)) {
    corr <- as_SFBM(corr0, file.path(tmp, "corr"), compact = TRUE)
  } else {
    corr$add_columns(corr0, nrow(corr))
  }
  log_msg("chromosome", chr, ":", length(cols), "SNPs")
}

for (label in labels) {
  df_beta <- matched[[label]]
  chi2 <- (df_beta$beta / df_beta$beta_se)^2
  ldsc <- snp_ldsc(ld, length(ld), chi2 = chi2, sample_size = df_beta$n_eff, blocks = NULL,
                   ncores = ncores)
  h2_init <- max(ldsc[["h2"]], 1e-3)
  log_msg(label, ": LDSC h2 =", ldsc[["h2"]], ", intercept =", ldsc[["int"]])
  # Keep the chains that converged (the bigsnpr tutorial's rule); if none does,
  # once more without the MLE update of p and h2 (bigsnpr's older, sturdier sampler).
  for (use_mle in c(TRUE, FALSE)) {
    multi <- snp_ldpred2_auto(corr, df_beta, h2_init = h2_init,
                              vec_p_init = seq_log(1e-4, 0.2, length.out = N_CHAINS),
                              allow_jump_sign = FALSE, shrink_corr = 0.95, use_MLE = use_mle,
                              ncores = ncores)
    spread <- sapply(multi, function(a) diff(range(a$corr_est)))
    keep <- which(spread > 0.95 * quantile(spread, 0.95, na.rm = TRUE))
    log_msg(label, ": use_MLE", use_mle, ": chain spreads", paste(signif(spread, 3), collapse = " "))
    log_msg(label, ": chain h2", paste(signif(sapply(multi, function(a) a$h2_est), 3),
                                       collapse = " "))
    if (length(keep) > 0) break
  }
  if (length(keep) == 0) stop(label, ": no LDpred2-auto chain converged")
  beta <- rowMeans(do.call(cbind, lapply(multi[keep], function(a) a$beta_est)))
  log_msg(label, ":", length(keep), "of", N_CHAINS, "chains kept")

  prefix <- file.path(fold_dir, paste0(label, ".ldpred2", tag))
  fwrite(data.frame(ID = map$rsid[ids], A1 = map$a1[ids], BETA = beta),
         paste0(prefix, ".betas.tsv"), sep = "\t")
  info <- c(
    snps = length(ids), qc_removed = sum(bad), window_kb = WINDOW_KB, use_mle = use_mle,
    chains = N_CHAINS, chains_kept = length(keep),
    ldsc_h2 = ldsc[["h2"]], ldsc_intercept = ldsc[["int"]],
    h2_est = mean(sapply(multi[keep], function(a) a$h2_est)),
    p_est = mean(sapply(multi[keep], function(a) a$p_est)),
    nonzero_betas = sum(beta != 0)
  )
  fwrite(data.frame(key = names(info), value = unname(info)), paste0(prefix, ".info.tsv"),
         sep = "\t")
}
