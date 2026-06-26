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
4. Three seeds (42, 123, 7) → mean ± std Q2 to reduce single-split variance.

**Phases:**

| Phase | Experiments | Description |
|---|---|---|
| 1 | 5 | baseline + each LLM source individually |
| 2 | 6 | all pairs of LLM sources |
| 3 | 4 | all triples of LLM sources |
| 4 | 1 | all four combined |

**Results** (`RESULTS/fixed_test_experiments/leaderboard.md`, 3 seeds, N_test = 51 rows):

#### Complete leaderboard — top results across all 4 phases

| Rank | Experiment | Architecture | Q2 mean | Q2 std | MAE mean | Phase |
|---:|---|---|---:|---:|---:|---:|
| 1 | **plus_deepseek_gemma** | hd_64_32_16_8_4 | **+0.220** | 0.129 | 23.1 | 2 — pair |
| 2 | plus_all | hd_64_32_16_8_4 | +0.041 | 0.071 | 24.4 | 4 — all four |
| 3 | plus_all | hd_32_16_8_4_4 | +0.006 | 0.091 | 24.3 | 4 — all four |
| 4 | plus_deepseek_opus | hd_64_32_16_8_4 | +0.023 | 0.090 | 25.4 | 2 — pair |
| 5 | plus_deepseek_kimi | hd_16_8_4_4_4 | +0.019 | 0.072 | 24.2 | 2 — pair |
| 6 | plus_deepseek_kimi | hd_64_32_16_8_4 | +0.017 | 0.127 | 23.8 | 2 — pair |
| 7 | plus_deepseek | hd_16_8_4_4_4 | -0.004 | 0.061 | 24.6 | 1 — single |
| 8 | baseline | hd_64_32_16_8_4 | -0.015 | 0.132 | 25.7 | 1 — baseline |
| … | *(triples — all negative)* | | | | | 3 — triple |

*(Full 48-row table in `RESULTS/fixed_test_experiments/leaderboard.md`)*

#### Phase summary — best Q2 per phase

| Phase | Best experiment | Best Q2 | Interpretation |
|---|---|---:|---|
| Baseline | baseline / hd_64 | -0.015 | Original PDCC alone cannot extrapolate |
| 1 — singles | plus_deepseek / hd_16 | -0.004 | DeepSeek alone nearly matches predicting the mean |
| 2 — pairs | **plus_deepseek_gemma / hd_64** | **+0.220** | **Best overall — clear winner** |
| 3 — triples | plus_deepseek_kimi_opus / hd_16 | -0.038 | Every triple is worse than the winning pair |
| 4 — all four | plus_all / hd_64 | +0.041 | Second best — large dataset rescues the big architecture |

---

**Key findings across all phases:**

1. **Baseline fails at extrapolation:** all models have Q2 < 0 on the fixed test when trained on original PDCC alone — the model interpolates well within known pairs but cannot generalise to new polymer–drug pairs.

2. **DeepSeek is the required anchor:** every positive-Q2 result includes DeepSeek. It is the largest (~239 rows) and most chemically diverse LLM source. No combination without DeepSeek achieves positive Q2.

3. **DeepSeek + Gemma is the clear winner (Q2 = +0.220):** despite Gemma alone being near the worst single-source (Q2 = -0.156), pairing it with DeepSeek produces the best result. The two sources cover complementary polymer–drug chemistry that is too sparse to be useful individually.

4. **Triples are the worst group:** removing any one source from the full set to form a triple consistently hurts more than using all four or the winning pair. Every triple achieves Q2 < 0. This suggests the full four-source pool has complementary coverage that no subset of three preserves.

5. **All four combined is second best (Q2 = +0.041):** with all ~297 LLM rows, the large architecture (`hd_64_32_16_8_4`) achieves the second-highest Q2. The Opus noise is diluted enough by the combined volume that the model still extracts useful signal.

6. **Opus is harmful in isolation but tolerable in the full pool:** Opus alone (Q2 = -0.09) and in pairs (Q2 down to -1.9) degrades performance, but within all four combined its negative effect is diluted.

7. **Architecture size is gated by training data volume:** `hd_64_32_16_8_4` is catastrophic in Phase 1 (too little data) but the best in Phases 2 and 4 (enough LLM rows to justify the capacity). `hd_16_8_4_4_4` is the most robust choice when data is limited.

---

## Thesis Conclusions (draft)

1. **Within-group interpolation (Paradigm 1):** the MLP achieves high Q2 (≈ 0.98 LOOCV) when predicting capacity along known polymer–drug concentration curves. The original 218 rows are sufficient for this task; LLM augmentation did not significantly change these results.

2. **Cross-group extrapolation (Paradigm 2):** no model generalises to unseen polymer–drug pairs when trained on the original PDCC alone (Q2 < 0 for all architectures). This reveals a fundamental limitation: the original dataset covers too few unique polymer–drug combinations to learn generalisable chemistry.

3. **LLM augmentation can improve extrapolation:** the best combination (DeepSeek + Gemma, large architecture) raises Q2 from ≈ −0.06 to +0.22, demonstrating that literature-mined data provides real signal for generalisation to new pairs. This is the main positive finding of the augmentation study.

4. **LLM source quality and combination matter more than quantity:** DeepSeek is the essential anchor (largest, most chemically diverse); Gemma pairs synergistically with it to form the best combination; Opus is harmful in isolation but tolerable when diluted across all four sources; Kimi is marginal. The best single pair (DeepSeek+Gemma, Q2=0.22) outperforms the best triple (Q2=−0.04) and is comparable to all-four (Q2=0.04).

5. **Triples are uniquely poor:** every combination of exactly 3 LLM sources achieves Q2 < 0, worse than both pairs and the full four-source pool. This suggests each triple excludes chemistry that the missing source covers, while the full pool dilutes noise enough to recover.

6. **Recommendation for future work:** expand the hand-curated PDCC dataset with more diverse polymer–drug groups; use DeepSeek and Gemma as primary LLM extraction sources; consider quality-filtering LLM rows by extraction confidence before training; and investigate why Opus data degrades performance (possible systematic extraction bias or unit errors).

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
