![3.11](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-11.yml/badge.svg)

# lele-py-bioinformatics — Thesis Reference

ML pipeline predicting polymer–drug adsorption **CAPACITY** (mg/g).
Built for a master's thesis studying whether LLM-extracted literature data improves a small hand-curated dataset.

---

## Thesis Goal

> **Can LLM-scraped experimental data from open-access papers improve a neural-network predictor of polymer–drug adsorption capacity, and if so, which LLM source contributes the most?**

The predictor is a Multilayer Perceptron (MLP) trained on the PDCC dataset. Four LLMs (Claude Opus, DeepSeek, Kimi, Gemma) each analysed ~2 000 open-access papers to extract polymer–drug–capacity measurements. The thesis compares the baseline (original PDCC only) against every augmentation strategy.

---

## Datasets

### Original — PDCC (Polymer–Drug Concentration–Capacity)
- **File:** `DATASETS/PDCC/polymer_drug_concentration_capacity.csv`
- **Size:** 218 rows, ~70 unique polymer–drug groups (after name resolution)
- **Schema:** `POLYMER_USED, DRUG, WATER_PH, CONCENTRATION, CAPACITY, SOURCE`
- **Target:** `CAPACITY` (mg/g) — how much drug a polymer adsorbs per gram

### LLM-scraped augmentation data
All files live in `DATASETS/PDCC/paper_scraper/`. Each is the **conflict-free** subset: globally deduplicated on `(POLYMER_PSMILES, DRUG_SMILES)` so that combining them yields no repeated pairs.

| File | LLM source | ~Rows | Notes |
|------|-----------|-------|-------|
| `pdcc_opus_without_conflicts.csv` | Claude Opus | 49 | Cleanest; manually reviewed |
| `pdcc_deepseek_without_conflicts.csv` | DeepSeek | 239 | Largest single-model source |
| `pdcc_kimi_without_conflicts.csv` | Kimi | 8 | Thinned (DeepSeek wins shared pairs) |
| `pdcc_gemma4_image_without_conflicts.csv` | Gemma (image) | 1 | Thinned |
| **Combined pool** | all four | **~297** | Conflict-free training set |

Name resolution dictionaries: `paper_scraper_complete_psmiles.json` (96 entries) + `paper_scraper_complete_smiles.json` (101 entries), merged with the built-in dicts for all experiments.

---

## Feature Engineering

Every row is featurised into **644 numeric features**:

| Feature group | Dimensions | Source |
|---|---|---|
| Polymer P-SMILES fingerprint (Morgan, 256-bit) | 256 | RDKit via polymetrix |
| Drug SMILES fingerprint (Morgan, 256-bit) | 256 | RDKit |
| Polymer polymetrix descriptors (ALL) | ~100 | polymetrix |
| Drug logP, logD at experiment pH | 2 | RDKit + dimorphite_dl |
| Drug HOMO–LUMO gap, net charge | 2 | xTB (gracefully skipped if it fails) |
| CONCENTRATION, WATER_PH | 2 | raw numeric |

Rows with unresolvable polymer/drug names or invalid numeric fields are dropped before training. The featurisation is joblib-cached (`.cache_dir/`) — delete the cache only if you suspect stale results.

---

## Model

**Architecture:** MLP with softplus output activation (capacity is strictly positive) and MSE loss.

Three architecture sizes were compared:

| Name | Hidden layers |
|---|---|
| `hd_16_8_4_4_4` | [16, 8, 4, 4, 4] |
| `hd_32_16_8_4_4` | [32, 16, 8, 4, 4] |
| `hd_64_32_16_8_4` | [64, 32, 16, 8, 4] |

**Training hyperparameters** (all experiments): dropout=0.2, weight_decay=1e-4, lr=1e-3, epochs=1000, early-stop patience=100, batch_size=16, StandardScaler on X, MinMaxScaler on y.

---

## Evaluation — Two Paradigms

### Paradigm 1 — Grouped K-fold Cross-Validation (within-group interpolation)

**What it tests:** given partial concentration–capacity data for a known polymer–drug pair, can the model predict capacity at untested concentrations?

**Design:** groups = unique (polymer, drug) pairs; each fold holds out one entire group; the model never sees *any* point from the held-out pair during training. Falls back to LOGO (Leave-One-Group-Out) when fewer than 5 groups are present.

**Metric:** Q2 (R² on the held-out predictions, concatenated across folds)

**Results** (`RESULTS/thesis_experiments/` and older `RESULTS/mlp_experiments/`):

| Experiment | Validation | Best Q2 |
|---|---|---|
| LOOCV — original PDCC, hd_16_8_4_4_4 | LOOCV | **0.984** |
| Grouped CV — original PDCC | Grouped K-fold | *(see thesis_experiments leaderboard)* |
| Grouped CV — plus DeepSeek | Grouped K-fold | *(see thesis_experiments leaderboard)* |
| … | … | … |

> **Interpretation:** Q2 ≈ 0.98 under LOOCV is high but optimistic — it tests interpolation along a known concentration curve, not generalisation to new pairs. The grouped CV is more conservative and the correct metric for this paradigm.

### Paradigm 2 — Fixed-Test-Set Augmentation Study (cross-group extrapolation)

**What it tests:** can the model predict capacity for **entirely new** polymer–drug pairs it has never seen?

**Design:**
1. The original PDCC is split **once per seed** by polymer–drug groups (80% train groups, 20% test groups) — saved as `original_train_groups_s{seed}.csv` / `original_test_groups_s{seed}.csv`.
2. LLM data is **leakage-filtered**: any LLM row whose `(polymer, drug)` resolves to the same SMILES pair as a test group is removed before merging.
3. Every experiment trains on `original_train + optional_LLM_data` and evaluates on the **same fixed test set** — enabling direct apples-to-apples comparison.
4. Seeds (42, 123, 7) → mean ± std Q2 to reduce single-split variance. A Phase 5 validation extended to 8 seeds (adding 0, 1, 2, 5, 10) to check robustness of apparent positive results.

**Phases:**

| Phase | Experiments | Description |
|---|---|---|
| 1 | 5 | baseline + each LLM source individually |
| 2 | 6 | all pairs of LLM sources |
| 3 | 4 | all triples of LLM sources |
| 4 | 1 | all four combined |

**Results** (`RESULTS/fixed_test_experiments/leaderboard.md`, 3 seeds 42/123/7, N_test varies 18–51 by seed):

#### Complete leaderboard — top results across all 4 phases (3-seed run)

| Rank | Experiment | Architecture | Q2 mean | Q2 std | MAE mean | Phase |
|---:|---|---|---:|---:|---:|---:|
| 1 | plus_deepseek_gemma ⚠ | hd_64_32_16_8_4 | +0.220 | 0.129 | 23.1 | 2 — pair |
| 2 | plus_all | hd_64_32_16_8_4 | +0.041 | 0.071 | 24.4 | 4 — all four |
| 3 | plus_all | hd_32_16_8_4_4 | +0.006 | 0.091 | 24.3 | 4 — all four |
| 4 | plus_deepseek_opus | hd_64_32_16_8_4 | +0.023 | 0.090 | 25.4 | 2 — pair |
| 5 | plus_deepseek_kimi | hd_16_8_4_4_4 | +0.019 | 0.072 | 24.2 | 2 — pair |
| 6 | plus_deepseek_kimi | hd_64_32_16_8_4 | +0.017 | 0.127 | 23.8 | 2 — pair |
| 7 | plus_deepseek | hd_16_8_4_4_4 | -0.004 | 0.061 | 24.6 | 1 — single |
| 8 | baseline | hd_64_32_16_8_4 | -0.015 | 0.132 | 25.7 | 1 — baseline |
| … | *(triples — all negative)* | | | | | 3 — triple |

⚠ **Phase 5 validation (8 seeds):** `plus_deepseek_gemma / hd_64` was retested with 5 additional seeds. All 5 new seeds gave negative Q2; the 8-seed mean drops to **−0.064** (std = 0.30). The initial +0.220 was a statistical artifact of 3 favorable splits. Per-seed breakdown:

| Seed | Q2 | N_test | Note |
|---:|---:|---:|---|
| 42 | +0.367 | 51 | original seed |
| 123 | +0.163 | 24 | original seed |
| 7 | +0.129 | 51 | original seed |
| 0 | −0.532 | 20 | validation seed |
| 1 | −0.057 | 24 | validation seed |
| 2 | −0.411 | 18 | validation seed |
| 5 | −0.076 | 20 | validation seed |
| 10 | −0.099 | 49 | validation seed |

*(Full 48-row table in `RESULTS/fixed_test_experiments/leaderboard.md`)*

#### Phase summary — best Q2 per phase (3-seed run; see ⚠ above for validated result)

| Phase | Best experiment | Best Q2 (3-seed) | 8-seed Q2 | Interpretation |
|---|---|---:|---:|---|
| Baseline | baseline / hd_64 | −0.015 | ≈ −0.4 | Original PDCC alone cannot extrapolate |
| 1 — singles | plus_deepseek / hd_16 | −0.004 | n/a | DeepSeek alone nearly matches predicting the mean |
| 2 — pairs | plus_deepseek_gemma / hd_64 | +0.220 ⚠ | **−0.064** | Not confirmed with more seeds |
| 3 — triples | plus_deepseek_kimi_opus / hd_16 | −0.038 | n/a | Every triple is worse than any pair |
| 4 — all four | plus_all / hd_64 | +0.041 | n/a | Marginal; high seed-to-seed variance |

---

**Key findings across all phases:**

1. **Baseline fails at extrapolation:** all models have Q2 < 0 on the fixed test when trained on original PDCC alone — the model interpolates well within known pairs but cannot generalise to new polymer–drug pairs.

2. **No augmentation strategy produces robust improvement:** the initial 3-seed result (DeepSeek+Gemma, Q2=+0.220) appeared positive, but Phase 5 validation with 8 seeds showed Q2=−0.064. The Q2 varies enormously across seeds (−0.53 to +0.37 for the same configuration), driven by which polymer–drug groups happen to fall in the test split — not by the augmentation strategy.

3. **DeepSeek is associated with the highest observed Q2:** every experiment that produced positive Q2 in the 3-seed run included DeepSeek (the largest, ~239 rows). No combination without DeepSeek exceeded the baseline in the initial study, consistent with its being the primary source of chemical diversity.

4. **Triples are the worst group in the initial study:** every combination of exactly 3 LLM sources achieved Q2 < 0 (3-seed run), worse than both pairs and the full four-source pool. This pattern was not confirmed with additional seeds but is noted as an exploratory observation.

5. **Architecture size is gated by training data volume:** `hd_64_32_16_8_4` is unstable in Phase 1 (too little data) and the most variable across seeds in all phases. `hd_16_8_4_4_4` is the most consistent choice when data is limited.

6. **The evaluation is fundamentally limited by dataset size:** ~70 polymer–drug groups in the PDCC yield ~14 test groups per seed. Q2 computed on ~14–51 test rows is highly sensitive to the specific groups selected, making any single-seed or small-sample estimate unreliable. More polymer–drug groups — not more LLM rows per group — are needed for a reliable extrapolation benchmark.

---

## Thesis Conclusions (draft)

1. **Within-group interpolation (Paradigm 1):** the MLP achieves high Q2 (≈ 0.98 LOOCV) when predicting capacity along known polymer–drug concentration curves. The original 218 rows are sufficient for this task; LLM augmentation did not significantly change these results.

2. **Cross-group extrapolation (Paradigm 2):** no model generalises to unseen polymer–drug pairs when trained on the original PDCC alone (Q2 < 0 for all architectures). This reveals a fundamental limitation: the original dataset covers too few unique polymer–drug combinations to learn generalisable chemistry.

3. **LLM augmentation did not produce a robust improvement:** the initial 3-seed study showed DeepSeek+Gemma reaching Q2=+0.220 — but Phase 5 validation with 8 seeds demonstrated this was a statistical artifact. The 8-seed mean for the same configuration is −0.064 (std = 0.30). No augmentation strategy achieves consistently positive Q2 across varied test splits.

4. **The bottleneck is the number of polymer–drug groups, not the number of rows per group:** Q2 is computed over ~14–51 test rows (14 test groups × variable concentration points). The high inter-seed variance (Q2 from −2.8 to +0.37 depending on the split) shows that the result depends on which groups land in the test set, not on model quality. Expanding the LLM corpus does not address this; more diverse hand-curated polymer–drug pairs do.

5. **DeepSeek is the most useful LLM source:** it is the largest (~239 rows) and most chemically diverse. All experiments that produced the highest (though not robust) Q2 values in the 3-seed study included DeepSeek. Opus degrades performance in small combinations; Kimi and Gemma contribute negligible novel pairs after deduplication.

6. **Recommendation for future work:** expand the hand-curated PDCC with more unique polymer–drug pairs (target >200 distinct groups) to make the fixed-test evaluation statistically meaningful; re-run the augmentation study under this larger baseline before drawing conclusions about LLM data quality; consider using Bayesian or ensemble methods that report calibrated uncertainty instead of single-point Q2.

---

## How to Reproduce

### Setup
```bash
# Install pixi: https://pixi.prefix.dev/latest/installation/
git clone https://github.com/Uomocosa/lele-py-bioinformatics
cd lele-py-bioinformatics
pixi install -e cuda   # or -e cpu for CPU-only
```

### Grouped CV experiments (Paradigm 1)
```bash
pixi run -e cuda run_integrated_paper_scraper_experiments
# Results → RESULTS/thesis_experiments/
```

### Fixed-test augmentation study (Paradigm 2)
```bash
# Full sweep — all 4 phases, all seeds
./unisi_scripts/07_run_fixed_test_experiments.sh

# Or phase-by-phase:
./unisi_scripts/07_run_fixed_test_experiments.sh --phase 1
./unisi_scripts/07_run_fixed_test_experiments.sh --phase 2
./unisi_scripts/07_run_fixed_test_experiments.sh --phase 3
./unisi_scripts/07_run_fixed_test_experiments.sh --phase 4

# Results → RESULTS/fixed_test_experiments/leaderboard.md
```

### On the UNISI server
Run scripts in order:
```bash
./unisi_scripts/00_bootstrap.sh           # first time only
./unisi_scripts/01_smoke_test.sh          # sanity check
./unisi_scripts/03_run_experiments.sh     # Paradigm 1
./unisi_scripts/07_run_fixed_test_experiments.sh  # Paradigm 2
./unisi_scripts/05_save_results.sh        # archive for scp
```

### Rebuild leaderboards from existing results
```bash
# Paradigm 1
pixi run -e cuda python -c "from bio.run_all_integrated_paper_scraper_experiments import rank; rank()"

# Paradigm 2
pixi run -e cuda python -c "from bio.run_fixed_test_experiments import rank; rank()"
```

---

## Repository Structure

```
bio/                          # main library (lazy-import system)
  Dataset/
    PDCC.py                   # dataset class: load → increment → resolve → featurize
    PDCCMethod/               # stateless functions for each pipeline step
    TorchDataset/PDCCtorch.py # PyTorch Dataset wrapper
    Splitted.py               # train/val/test split + scale()
  ML/
    MLP.py                    # MLP model + Config
    MLPMethod/                # train_model, save_model, check_accuracy
  mlp_experiment/             # old LOOCV experiment runner
  integrate_paper_scraper.py  # CLI to merge and featurise LLM-scraped CSVs
  run_all_integrated_paper_scraper_experiments.py   # Paradigm 1 runner
  run_fixed_test_experiments.py                     # Paradigm 2 runner

DATASETS/PDCC/
  polymer_drug_concentration_capacity.csv           # original 218-row PDCC
  paper_scraper/                                    # LLM-scraped CSVs + name dicts

RESULTS/
  thesis_experiments/         # Paradigm 1 output (grouped CV leaderboard + plots)
  fixed_test_experiments/     # Paradigm 2 output (fixed-test leaderboard)
  mlp_experiments/            # old LOOCV results (kept for reference)

unisi_scripts/                # server bootstrap + run scripts (00–07)
```

---

## Notes for Thesis Writing

- **Q2** in this project = R² computed on the concatenated held-out predictions (same as the Q² metric used in QSAR literature). Range: (−∞, 1]. Q2 = 0 means predicting the training mean for all points.
- **Grouped CV vs LOOCV:** LOOCV (older results, Q2 ≈ 0.98) tests interpolation on *augmented* data (includes interpolated concentration points) — inflate scores. Grouped CV tests on *original* data points only — more honest for Paradigm 1.
- **Fixed-test negative Q2 ≠ model broken:** it means extrapolation to new polymer–drug pairs is hard, not that the model is wrong. This is a known challenge in QSAR with small, structurally diverse datasets.
- **LLM data row counts after filtering:** deepseek contributes ~239 rows; after leakage filtering and deduplication the effective addition varies per seed. The `all_runs.csv` detail file records `N_test` per run; consult `_data/` in the results dir for the materialised training CSVs.
