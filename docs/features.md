# SNP features for the auxiliary network

The auxiliary network turns per-SNP features into the main network's first
layer. The published model uses each SNP's genotype frequencies per population
(`dietnet generate-embedding`). Upstream's model reads its auxiliary input from
`embedding.npz`: a SNPs × features matrix, one per fold (3-D) or shared by all
folds (2-D), whose columns it L2-normalises itself. Anything that can be written
as such a matrix can feed the model, without changing upstream's code.

`dietnet build-embedding` writes one from *blocks* described in a YAML spec:

```yaml
# features.yaml
blocks:
  - provider: class_genotype_freq      # the published features (upstream's script)
  - provider: precomputed              # any per-SNP table: chrom, pos, ref, alt, features...
    path: scores.tsv
    orient: effect                     # ALT-vs-REF values: flipped where A1 is REF
    pca: 64                            # keep this block to 64 columns
  - provider: random                   # control: same width, no information
    columns: 64
```

```bash
dietnet build-embedding --dataset dataset.h5 --partition partition.npz \
                        --spec features.yaml --out embedding.npz
```

Then train with `--embedding embedding.npz` as usual. The command prints the
columns each block occupies, and writes `embedding.npz.json` beside the matrix
describing every block and column; `train` copies that description into each
package's `metadata.json` (as `embedding`), so a package records the features
it was trained on.

## Providers

`class_genotype_freq`
:   Genotype frequencies per class and fold: upstream's `generate_embedding.py`,
    run unmodified. Alone, the result is exactly what
    `dietnet generate-embedding` writes. Takes `ncpus`.

`precomputed`
:   A table with `chrom`, `pos`, `ref`, `alt` and numeric feature columns (TSV,
    optionally gzipped, or parquet, which needs `pyarrow` installed), matched to the dataset's SNPs by position
    and alleles, on either strand. `columns:` picks feature columns (default:
    all the others); `name:` prefixes them (default: the file's stem).
    `path:` is relative to the directory `dietnet` runs in.
    Missing values (NaN) are refused: fill or drop them first.

    - `orient: effect`: each value describes ALT relative to REF, so it is
      negated where the dataset counts REF. plink 1.9 makes the minor allele
      A1, which is REF for a quarter of the 1KGP SNPs. A/T and C/G SNPs fit
      both orientations, so their sign cannot be known and they are treated as
      missing.
    - `orient: none` (the default): values are used as they are.

    SNPs the table lacks get zeros and are flagged in an added
    `<name>:missing` column.

`random`
:   `columns:` standard-normal columns (`seed:`, default 0): the control for
    "more columns" rather than "more information".

`shuffled`
:   Another block with its SNP rows permuted. `of:` is that block's index in
    the list (0 for the first), `seed:` the permutation's seed: the control for
    "this information" rather than "information of this shape".

## Transforms

Applied to a block after its provider:

- `center: true` subtracts each column's mean over SNPs.
- `pca: k` keeps the first k principal components over SNPs, per fold for a
  per-fold block.

Upstream's per-column L2 normalisation undoes any per-column scaling, so these
are the transforms that matter: centring, and controlling how many columns,
and so how much weight, a block brings.

A fold-independent block is repeated for every fold when combined with a
per-fold one. Every column must vary: an all-zero column would be divided by a
zero norm inside the model and turn it to NaN, so building refuses one.

## AlphaGenome variant effects

[AlphaGenome](https://github.com/google-deepmind/alphagenome) predicts many
genomic tracks from DNA sequence, and scores a variant by how its ALT allele
changes them relative to REF. `dietnet alphagenome` scores every SNP of a
dataset into a table for the `precomputed` provider.

!!! warning "This sends your variants to Google"
    Scoring goes through Google DeepMind's **hosted** AlphaGenome API: the
    chromosome, position, REF and ALT of every SNP in the dataset are sent to
    it. No genotypes or sample ids are sent, but check that sharing the variant
    list is allowed for your data. The API needs a personal key, and its terms
    of use are non-commercial; read them before you use it.

```bash
uv sync --frozen --extra alphagenome      # or: pip install -e '.[alphagenome]'
export ALPHAGENOME_API_KEY=...            # never in a file or config
dietnet alphagenome --dataset dataset.h5 --out alphagenome.tsv
```

The key is read only from `ALPHAGENOME_API_KEY`. Each output row is a SNP
(`chrom`, `pos`, `ref`, `alt`) and one column per track,
`<output_type>|<track_name>`, holding the signed ALT − REF `raw_score`.
Gene-level scorers report a score per nearby gene; the gene with the largest
effect on the track is kept. A variant with no score for a track (no gene in
range) gets 0.

| option | default | |
|---|---|---|
| `--scorers` | `ATAC,DNASE,CHIP_TF,CHIP_HISTONE,CAGE,RNA_SEQ` | AlphaGenome's recommended scorers, by name |
| `--sequence-length` | `1MB` | context around each variant: `16KB`, `100KB`, `500KB` or `1MB` |
| `--chunk` | 200 | SNPs per saved part |
| `--max-workers` | 5 | parallel API requests |

**Resumable.** Each chunk is written to `alphagenome.tsv.parts/` as soon as it
arrives. Hundreds of thousands of API calls will be interrupted at some point;
rerun the same command and only the SNPs no part holds yet are scored. The
final table is assembled from the parts in the dataset's SNP order.

**SNP ids must be `chr:pos:REF:ALT` on GRCh38**, as in the 1KGP data. A
`.bim`'s A1/A2 say which allele is counted, not which is the reference, so REF
and ALT have to come from the id; the command stops if they cannot.

Use it as a block:

```yaml
  - provider: precomputed
    path: alphagenome.tsv
    orient: effect     # flipped where the dataset counts REF (a quarter of the 1KGP SNPs)
    pca: 64            # thousands of tracks -> 64 columns
```

For an experiment, compare it against the same spec with a `random` block of
the same width, and with a `shuffled` copy of the AlphaGenome block, and
against the MLP baseline (`dietnet train --model mlp`).

!!! note
    The AlphaGenome step is tested against a stand-in for the API and against
    the real client's types, but has not yet been run against the API itself
    (see the [plan](refactor-plan.md)).
