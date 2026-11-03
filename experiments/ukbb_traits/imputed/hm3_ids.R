# HapMap3+ variants (bigsnpr's map_hm3_plus.rds, GRCh38 positions) matched to the
# imputed genotypes' bims, one id list per chromosome.
#
#   Rscript hm3_ids.R <map_hm3_plus.rds> <imputed prefix with {chr}> <out dir>
#
# Matched by chromosome, GRCh38 position and the two alleles in either order (both
# come from UK Biobank's imputation on the + strand, so no strand flips). Writes
# <out dir>/chr<c>.ids and <out dir>/summary.tsv (HM3+ variants, matched, per chr).
suppressPackageStartupMessages(library(data.table))
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
map <- as.data.table(readRDS(args[1]))[!is.na(pos_hg38)]
dir.create(args[3], showWarnings = FALSE, recursive = TRUE)

summary <- rbindlist(lapply(1:22, function(chr) {
  bim <- fread(paste0(sub("{chr}", chr, args[2], fixed = TRUE), ".bim"), header = FALSE,
               select = c(2, 4, 5, 6), col.names = c("id", "pos", "b1", "b2"),
               colClasses = list(character = c(2, 5, 6), integer = 4))
  hm3 <- map[map$chr == chr, .(pos = pos_hg38, a0, a1)]
  m <- merge(bim, hm3, by = "pos")
  m <- m[(b1 == a1 & b2 == a0) | (b1 == a0 & b2 == a1)]
  ids <- unique(m$id)
  fwrite(list(ids), file.path(args[3], paste0("chr", chr, ".ids")), col.names = FALSE)
  cat("chromosome", chr, ":", nrow(hm3), "HM3+ variants,", length(ids), "matched\n")
  data.table(chr = chr, hm3 = nrow(hm3), matched = length(ids))
}))
fwrite(summary, file.path(args[3], "summary.tsv"), sep = "\t")
cat("total:", sum(summary$hm3), "HM3+ variants,", sum(summary$matched), "matched\n")
