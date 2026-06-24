# Plan: Integrate `paper-scraper` output into `lele-py-bioinformatics` (`bio`)

> **Audience:** an agent working **inside the `lele-py-bioinformatics` repo** that has no prior
> context on either project. Read this top to bottom before touching code. Every path, command,
> and API you need is here.

---

## 1. TL;DR — what you are building

`paper-scraper` is a sibling project (at `../paper-scraper`, i.e.
`C:\Users\SamueleMaggiori\paper-scraper`) that mines scientific papers and produces a dataset of
**polymer–drug adsorption measurements** plus resolved chemical structures. This repo (`bio`) is the
ML library that consumes that data to predict adsorption **CAPACITY**.

Your job:

1. Add a small, well-documented **entry point (CLI)** to `bio` that loads one or more of
   paper-scraper's CSV datasets, resolves chemical names to structures using paper-scraper's
   dictionaries, runs the existing `bio` featurization pipeline, and produces a featurized,
   train-ready dataset (and optionally trains/evaluates a model).
2. Fix **two real bugs** in `bio` that silently drop good rows.
3. Verify end-to-end against the expected numbers in §8.

This has already been **validated as feasible** — the data contract matches and name resolution
covers ~98–100% of rows (see §8 for measured numbers). You are productionizing a proven path, not
exploring an unknown one.

**Out of scope:** improving the data itself (more polymers resolved, cleaning range values). That is
a separate, ongoing effort in `paper-scraper`.

---

## 2. Background — the two projects and the data

- **Thesis goal:** discover polymers that capture water pollutants (drugs/dyes/metals). The model
  learns to predict how much of a drug a polymer adsorbs (`CAPACITY`, mg/g) given conditions.
- **paper-scraper** does: paper mining → LLM extraction → chemical-name resolution (drug → SMILES,
  polymer → P-SMILES) → emits clean CSVs + JSON name→structure maps.
- **`bio`** does: name→structure lookup → molecular featurization (RDKit, polymetrix,
  dimorphite_dl) → PyTorch dataset → model training.

### The data contract (PDCC = Polymer–Drug–Concentration–Capacity)

paper-scraper emits **6-column CSVs**, which is exactly the schema `bio` already reads
(`bio/__global__.py` → `PDCC_DATASET`):

| Column          | Meaning                                   | Used by featurization? |
|-----------------|-------------------------------------------|------------------------|
| `POLYMER_USED`  | polymer **name** (later → P-SMILES)       | yes (after resolution) |
| `DRUG`          | drug/molecule **name** (later → SMILES)   | yes (after resolution) |
| `WATER_PH`      | pH of the water (float)                   | yes (logD, net charge) |
| `CONCENTRATION` | initial concentration (float)             | yes (feature)          |
| `CAPACITY`      | adsorption capacity — **the target label**| yes (label)            |
| `SOURCE`        | citation/URL string                       | **no — metadata only** |

paper-scraper also emits two **name→structure dictionaries** (JSON, `{name: structure}`):

- `paper_scraper_complete_psmiles.json` — 212 polymer names → P-SMILES
- `paper_scraper_complete_smiles.json` — 124 drug names → SMILES

`bio` resolves names→structures via dictionaries it already supports overriding (see §4). So the
integration is: **feed paper-scraper's CSV as the dataset, and inject paper-scraper's two JSON dicts
as the lookup tables.**

### Where to find paper-scraper's files

**Use the conflict-free set in `../paper-scraper/output_filtered/`** (not the raw
`output/` files). paper-scraper deduplicates `(POLYMER_PSMILES, DRUG_SMILES)` tuples
**globally across all the per-model CSVs**: each tuple is kept only in the best
model's file under a single paper (`opus > deepseek > kimi > gemma`), and removed
from the others. This matters because `bio` concatenates these CSVs and has **no
PAPER column** to deduplicate itself — so the same polymer–drug pair would otherwise
appear under several papers across files. With the filtered set, **concatenating all
the per-model files yields a conflict-free training set**:

```
pdcc_opus_without_conflicts.csv                    ~49 rows  — manually reviewed (cleanest)
pdcc_deepseek_without_conflicts.csv               ~239 rows  — largest single-model
pdcc_kimi_without_conflicts.csv                     ~8 rows  — thinned: deepseek wins shared tuples
pdcc_gemma4_image_without_conflicts.csv             ~1 row   — thinned
pdcc_gemma4_text_without_conflicts.csv               0 rows  — empty, ignore
pdcc_matched_deepseek_kimi_without_conflicts.csv   ~93 rows  — STANDALONE gold set (see note)
paper_scraper_complete_psmiles.json
paper_scraper_complete_smiles.json
removed_rows_report.csv                                      — audit of what was dropped
```

There is **no combined `pdcc_deepseek_kimi_gemma` file** — you assemble the combined
set yourself by passing all the per-model files to `--pdcc-datasets` (see §4). The
lower-priority files are intentionally thinned (the data isn't lost — the winning
tuple lives in the higher-priority file).

> **`pdcc_matched_*` is NOT part of the combine pool.** It is the deepseek∩kimi
> agreement subset, so its rows duplicate deepseek/kimi by construction. It is deduped
> only within itself and meant for **separate evaluation**, not for concatenating with
> the single-model files. Do not pass it alongside deepseek/kimi or you reintroduce
> duplicate tuples.

The user will **copy these into this repo by hand.** Standardize on this target location and write
your code to read from it:

```
lele-py-bioinformatics/DATASETS/PDCC/paper_scraper/
    pdcc_*_without_conflicts.csv
    paper_scraper_complete_psmiles.json
    paper_scraper_complete_smiles.json
```

(If the folder is missing, print a clear error telling the user to copy the files there.)

---

## 3. How this repo works (orientation — read before coding)

- **Package manager:** [pixi](https://pixi.sh). Config in `pixi.toml`. `bio` is installed
  **editable** (`pyproject.toml` is `bio = { path = ".", editable = true }`), so editing files under
  `bio/` takes effect immediately — no reinstall.
- **Run anything with:** `pixi run python ...` (from the repo root). The default environment is
  `["cpu"]` and **includes CPU PyTorch**, RDKit, polymetrix, dimorphite_dl, etc.
- **Custom import system:** this repo uses `new_import_system` (a dependency). Submodules are
  **lazy-loaded** — e.g. `bio.Dataset.PDCC`, `bio.Dataset.PDCCMethod.featurize`,
  `bio.Metric.calculate_logp` resolve on attribute access. `bio/Dataset/__init__.py` is essentially
  empty on purpose; do not "fix" it by adding explicit imports.
- **Import-time side effects:** `bio/__global__.py` asserts `DATASETS_DIR`, `VOCABULARIES_DIR`,
  `PDCC_DATASET` etc. exist. Inside this repo they do, so `import bio` works. (Note: a *non-editable*
  copy of `bio` loses `DATASETS/` and fails this assertion — that only bit the paper-scraper env, not
  you. Stay in this repo and you are fine.)
- **Caching:** `convert_names_to_smiles` and `featurize` are decorated with
  `@CACHE_MEMORY.cache` (joblib, cache dir `.cache_dir`). joblib keys on function source + args, so
  your code edits auto-invalidate. If you ever see stale results, delete `.cache_dir`.

### The existing pipeline (the classes you will drive)

`bio/Dataset/PDCC.py`:

```python
@dataclass
class Config:
    csv_file: Path
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None
    seed: int = 42

class PDCC:
    def __init__(self, config: Config): self.df = pd.read_csv(config.csv_file); ...
    def increment_dataset(self, options=...): ...        # optional data augmentation
    def convert_names_to_smiles(self, options=...): ...  # NAME -> structure (in place on self.df)
    def to_torch_dataset(self) -> torch.utils.data.Dataset: ...  # featurizes + builds tensors
```

`bio/Dataset/PDCCMethod/convert_names_to_smiles.py`:

```python
@dataclass
class Options:
    psmiles_dict: dict = field(default_factory=lambda: PSMILES_DICT)  # <-- override this
    smiles_dict:  dict = field(default_factory=lambda: SMILES_DICT)   # <-- and this

def convert_names_to_smiles(df, options=Options()):
    # case-insensitive map of POLYMER_USED -> psmiles, DRUG -> smiles; warns on misses
    ...
```

`bio/Dataset/TorchDataset/PDCCtorch.py` — `to_torch_dataset()` returns this. It requires columns
`POLYMER_USED, DRUG, CONCENTRATION, CAPACITY`, calls `featurize`, then builds
`X = numeric columns except CAPACITY` and `y = CAPACITY`. (String `SOURCE` is naturally excluded from
`X` because it is non-numeric.)

`bio/Dataset/PDCCMethod/featurize.py` — produces ~1000+ feature columns (`poly_*`, `drug_*`,
fingerprint bits) from RDKit + polymetrix + dimorphite_dl.

`bio/Dataset/split_dataset.py` (used in `PDCC.test_usage`) — splits a torch dataset and offers
`.scale(...)` with a sklearn scaler. Reuse it; don't reimplement splitting.

The canonical end-to-end usage already lives in `PDCC.py::test_usage()` — read it; your CLI is a
parameterized version of it.

---

## 4. Implementation

### Step 0 — Environment

```bash
pixi install            # materialize the default (cpu) env; brings in torch, rdkit, polymetrix, ...
pixi run python -c "import bio; import torch; print('ok', torch.__version__)"
```

> Windows note: `pixi install` can fail with "Access is denied" on a wheel if a file is locked by
> another process (e.g. an editor, a running python, antivirus). Close other processes and retry;
> use `pixi run --frozen ...` to run without re-syncing once the env exists.

### Step 1 — Fix the two row-dropping bugs (do this first; it changes the numbers)

Both functions call a **blanket `df.dropna()`** that drops a row if *any* column is NaN — including
`SOURCE`, which is pure metadata and not used for training. This silently throws away good rows.

**Bug A — `bio/Dataset/PDCCMethod/convert_names_to_smiles.py` (~line 39):**

```python
# before
df = df.dropna()
# after — only drop rows whose name resolution actually failed
df = df.dropna(subset=['POLYMER_USED', 'DRUG'])
```

**Bug B — `bio/Dataset/PDCCMethod/featurize.py` (~line 110):**

```python
# before
df = df.dropna()
# after — drop rows with NaN in any feature/label/condition column, but ignore metadata (SOURCE)
metadata_cols = ['SOURCE']
feature_cols = [c for c in df.columns if c not in metadata_cols]
df = df.dropna(subset=feature_cols)
```

Rationale & evidence: `pdcc_gemma4_image.csv` resolves **every** name (13/13 polymers, 17/17 drugs)
yet currently yields **0 rows**, solely because its `SOURCE` column is empty in all 43 rows. After
this fix those rows survive. The other datasets are unaffected (their `SOURCE` is populated).

> Keep `WATER_PH`, `CONCENTRATION`, `CAPACITY` in the dropna subset — a row missing any of those
> genuinely cannot be featurized or trained on and *should* drop.

### Step 2 — Build the integration entry point (CLI)

Create `bio/integrate_paper_scraper.py`. Use **`tyro`** for the CLI (already a dependency; the repo
uses it elsewhere). Desired interface (the user explicitly asked for list-style flags):

```
# Combined conflict-free training set = pass ALL the per-model pool files:
pixi run python -m bio.integrate_paper_scraper \
    --pdcc-datasets DATASETS/PDCC/paper_scraper/pdcc_opus_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_deepseek_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_kimi_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_gemma4_image_without_conflicts.csv \
    --psmiles-dicts DATASETS/PDCC/paper_scraper/paper_scraper_complete_psmiles.json \
    --smiles-dicts  DATASETS/PDCC/paper_scraper/paper_scraper_complete_smiles.json
```

> Pass the per-model pool files **all together** — that is how the combined set is
> formed (there is no pre-combined file), and the global dedup guarantees no
> `(PSMILES, SMILES)` tuple repeats across them. Do **not** add `pdcc_matched_*` here
> (standalone subset — see §2).

Behavior:

1. **Load & merge datasets** — read each `--pdcc-datasets` CSV, `pd.concat` them. Optionally
   de-duplicate on `(POLYMER_USED, DRUG, WATER_PH, CONCENTRATION, CAPACITY)`. Skip empty files
   (e.g. `pdcc_gemma4_text.csv`).
2. **Load & merge dicts** — read each `--psmiles-dicts` / `--smiles-dicts` JSON
   (`{name: structure}`). Merge into one dict each; on key collision, later file wins; drop
   empty/`"NOT_A_VALID_POLYMER"` values. (Default to the two paper_scraper_complete_*.json files if
   no flag is passed.)
3. **Resolve** — build
   `PDCCMethod.convert_names_to_smiles.Options(psmiles_dict=merged_psmiles, smiles_dict=merged_smiles)`
   and run the pipeline:

   ```python
   from pathlib import Path
   import bio
   from bio.Dataset import PDCC, PDCCMethod

   config = PDCC.Config(csv_file=<combined_csv_or_tmp>)   # PDCC reads a CSV path; see note below
   ds = PDCC.PDCC(config)
   # (optional) ds.increment_dataset(...)  # only if you want interpolation/origin augmentation
   ds.convert_names_to_smiles(
       PDCCMethod.convert_names_to_smiles.Options(
           psmiles_dict=merged_psmiles, smiles_dict=merged_smiles
       )
   )
   torch_ds = ds.to_torch_dataset()       # featurizes; returns PDCCtorch
   ```

   > `PDCC.Config.csv_file` takes a path, not a DataFrame. If you merged multiple CSVs in memory,
   > write the combined frame to a temp CSV (or `DATASETS/PDCC/paper_scraper/_combined.csv`) and
   > point `Config` at it. Alternatively, instantiate `PDCC` then overwrite `ds.df` with your merged
   > frame before calling `convert_names_to_smiles` — pick one and document it.

4. **Report** — log: rows in, rows after resolution, rows after featurization, `num_features`, and
   the set of names that failed to resolve (so the user knows what to add to the dicts).
5. **Output** — save the featurized dataset (e.g. to `DATASETS/PDCC/paper_scraper/featurized.csv`)
   and/or split + scale for training:

   ```python
   split = bio.Dataset.split_dataset(
       dataset=torch_ds, train_percentage=0.6, validation_percentage=0.2,
       test_percentage=0.2, seed=config.seed,
   )
   from sklearn.preprocessing import StandardScaler
   split.scale(feature_col_indexes=range(torch_ds.num_features), scaler_fn=StandardScaler())
   ```

Keep the module importable and add a `main()` so it can also become a `pixi.toml` task later (mirror
the existing `[tasks]` entries, e.g. `integrate_paper_scraper = "python -m bio.integrate_paper_scraper"`).

### Step 3 (optional, recommended) — a sanity script / test

Add a quick check (or a pytest under `bio/`, respecting the repo's pytest markers) that runs the
pipeline on `pdcc_opus.csv` and asserts `num_features > 0` and surviving rows ≈ expected (§8). Mark
it `above10s` if it is slow, so it is skipped by the default `addopts`.

---

## 5. Reuse, don't reinvent

- Name resolution: **`bio.Dataset.PDCCMethod.convert_names_to_smiles`** with `Options` overrides.
- Featurization: **`bio.Dataset.PDCC.to_torch_dataset()`** (drives `featurize` + `PDCCtorch`). Do not
  call featurizers directly.
- Splitting & scaling: **`bio.Dataset.split_dataset`** + its `.scale(...)`.
- Reference flow to copy: **`bio/Dataset/PDCC.py::test_usage()`**.
- The standalone 10-stage validator already written in paper-scraper —
  `../paper-scraper/scripts/check_featurization_failures.py` — is a useful oracle to cross-check your
  surviving-row counts.

---

## 6. Files you will create / modify

| File | Action |
|------|--------|
| `bio/Dataset/PDCCMethod/convert_names_to_smiles.py` | edit dropna (Bug A) |
| `bio/Dataset/PDCCMethod/featurize.py` | edit dropna (Bug B) |
| `bio/integrate_paper_scraper.py` | **new** — the CLI entry point |
| `pixi.toml` `[tasks]` | (optional) add an `integrate_paper_scraper` task |
| `DATASETS/PDCC/paper_scraper/` | data dropped here by the user (you read from it) |

Do **not** edit `bio/__global__.py`'s `PSMILES_DICT` / `SMILES_DICT` — inject via `Options` instead,
so the integration stays non-invasive and reversible.

---

## 7. Known gotchas

- **Use `output_filtered/` (the `*_without_conflicts.csv` files), not raw `output/`.** The pool
  files are already globally deduplicated, so the combined set has no repeated `(PSMILES, SMILES)`
  tuple. Do not pass `pdcc_matched_*` alongside the pool (it is a deepseek∩kimi subset → reintroduces
  duplicates). There is no pre-combined file — concatenate the pool files via `--pdcc-datasets`.
- **`pdcc_gemma4_text_without_conflicts.csv` is empty (0 rows)** — skip it.
- **`pdcc_gemma4_image.csv` has empty `SOURCE`** — 0 survivors until you apply the Bug B fix; weakest
  dataset, lowest priority.
- **One corrupted polymer name** `PPy<U+FFFD>SD` (a U+FFFD replacement char from an encoding error)
  is the single unresolvable polymer across all datasets. Read JSON/CSV as UTF-8; expect this one to
  just drop.
- **Range strings** like `WATER_PH="6.6-6.8"` or `CONCENTRATION="50-900"` exist in a small fraction of
  rows and will fail numeric featurization (→ NaN → dropped). Pre-cleaning them to a representative
  value is optional and belongs to a later data-quality pass; paper-scraper's
  `check_featurization_failures.py` has the detection helpers (`has_range_string`).
- **joblib cache** (`.cache_dir`) — delete it if you ever suspect stale featurization results.
- **Heavy first run** — featurization (polymetrix + dimorphite_dl) on a few hundred rows takes a few
  minutes; this is normal.

---

## 8. Verification

Row counts are for the **conflict-free `output_filtered/` files** (see §2). The lower-priority
files are thinned because the global dedup gives each `(PSMILES, SMILES)` tuple to the best model;
the data is not lost, it lives in the higher-priority file. The combined pool (all four per-model
files) is **~297 rows with zero repeated tuples**. After resolution the survived count is ~1 lower
per file (one unresolvable polymer `PPy<U+FFFD>SD`); final featurized counts drop slightly more
where SMILES are invalid or pH/conc are ranges.

| dataset (`output_filtered/`) | rows | note |
|---|---|---|
| pdcc_opus_without_conflicts.csv | ~49 | cleanest |
| pdcc_deepseek_without_conflicts.csv | ~239 | largest; wins most shared tuples |
| pdcc_kimi_without_conflicts.csv | ~8 | thinned by deepseek |
| pdcc_gemma4_image_without_conflicts.csv | ~1 | thinned; needs Bug B fix to survive featurization |
| pdcc_gemma4_text_without_conflicts.csv | 0 | empty, ignore |
| **combined pool (opus+deepseek+kimi+gemma)** | **~297** | **conflict-free training set** |
| pdcc_matched_deepseek_kimi_without_conflicts.csv | ~93 | STANDALONE — do not concat with the pool |

**Verification commands:**

```bash
# 1. Smallest, cleanest dataset end-to-end:
pixi run python -m bio.integrate_paper_scraper \
    --pdcc-datasets DATASETS/PDCC/paper_scraper/pdcc_opus_without_conflicts.csv
# Expect: ~48 rows resolved, a featurized frame with num_features > 0 (hundreds/thousands), no crash.

# 2. Full combined conflict-free set (all per-model pool files together):
pixi run python -m bio.integrate_paper_scraper \
    --pdcc-datasets DATASETS/PDCC/paper_scraper/pdcc_opus_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_deepseek_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_kimi_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_gemma4_image_without_conflicts.csv \
    --psmiles-dicts DATASETS/PDCC/paper_scraper/paper_scraper_complete_psmiles.json \
    --smiles-dicts  DATASETS/PDCC/paper_scraper/paper_scraper_complete_smiles.json
# Expect: ~297 rows in, no duplicate (PSMILES, SMILES) tuple across the set.

# 3. Confirm gemma4_image survives featurization (Bug B regression check):
pixi run python -m bio.integrate_paper_scraper \
    --pdcc-datasets DATASETS/PDCC/paper_scraper/pdcc_gemma4_image_without_conflicts.csv
# Expect: > 0 surviving rows (was 0 before the fix).
```

**Success criteria:**

1. `pixi run python -c "import bio, torch"` works in the default env.
2. The CLI runs on every non-empty `pdcc_*.csv` without crashing and prints the in/resolved/featurized
   counts + unresolved-name report.
3. Surviving row counts match the table above (within a couple of rows); `gemma4_image` > 0 after the
   fix.
4. Output featurized CSV has a numeric `CAPACITY` label column and many `poly_*` / `drug_*` feature
   columns; `to_torch_dataset().num_features > 0`.
5. (If implemented) split + scale runs and produces train/val/test tensors.

---

## 9. Definition of done

- Two dropna bugs fixed.
- `bio/integrate_paper_scraper.py` exists, is documented, supports list-valued `--pdcc-datasets`,
  `--psmiles-dicts`, `--smiles-dicts`, merges them, runs the real `bio` pipeline, reports counts and
  unresolved names, and writes a featurized output.
- Verification §8 passes.
- No edits to `bio/__global__.py` dicts; integration is via `Options` injection.
