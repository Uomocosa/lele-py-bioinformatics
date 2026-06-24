![3.11](https://github.com/Uomocosa/lele-py-bioinformatics/actions/workflows/test-3-11.yml/badge.svg)

# How to use
1. Install [pixi](https://pixi.prefix.dev/latest/installation/)
2. `git clone https://github.com/Uomocosa/lele-py-bioinformatics`
3. `cd lele-py-bioinformatics`
3. `pixi install -e cpu` or you can try `pixi install -e cuda`.
    - ***Note!*** To make `pixi install -e cuda` work you might need to change the `pixi.toml`.
4. To test if it works run the following command:
    - (cpu) `pixi run find_polymer_for_target_molecule "aspirin"`
    - (cuda) `pixi run -e cuda find_polymer_for_target_molecule "aspirin"`
    - Also run `pixi run find_polymer_for_target_molecule --help` to see the options and default values.

# My personal bioinformatics library
- I used this library and apps for my master's thesis.
- It uses my [new-python-import-system](https://github.com/Uomocosa/new-python-import-system). The imports might seem more _magical_ than usual.

# Data and Outputs
- The **datasets** used can be found in the `DATASET/` folder.
- All the **results** can be found in the `RESULTS/` folder. Each should contain a README.md file explaining how it was found.

# Integrating `paper-scraper` data
[`paper-scraper`](https://github.com/Uomocosa/paper-scraper) mines papers and emits polymer–drug
adsorption datasets (`pdcc_*.csv`, the 6-column `POLYMER_USED, DRUG, WATER_PH, CONCENTRATION,
CAPACITY, SOURCE` schema) plus two name→structure dictionaries
(`paper_scraper_complete_psmiles.json`, `paper_scraper_complete_smiles.json`).

`bio.integrate_paper_scraper` loads one or more PDCC CSVs, resolves names to structures using
configurable dictionaries (injected via `Options`, so the in-code global dicts are never mutated),
runs the normal featurization pipeline, and writes a train-ready `featurized.csv`.

**Defaults:** with no flags it featurizes the **original** PDCC dataset
(`DATASETS/PDCC/polymer_drug_concentration_capacity.csv`) using the builtin global dicts:
```bash
pixi run integrate_paper_scraper
```

**Dictionary sources** (`--psmiles-dicts` / `--smiles-dicts`) are a list of either the `builtin`
token (the in-code global `PSMILES_DICT` / `SMILES_DICT`) or JSON file paths, merged in order
(later wins). Passing values **replaces** the default `["builtin"]`, so include `builtin` to keep
the global dict alongside paper-scraper's.

**Using paper-scraper data.** Put `paper-scraper`'s conflict-free `output_filtered/` files here:
```
DATASETS/PDCC/paper_scraper/
    pdcc_*_without_conflicts.csv          # the per-model pool
    paper_scraper_complete_psmiles.json
    paper_scraper_complete_smiles.json
```
Use the `*_without_conflicts.csv` set (globally deduplicated on `(P-SMILES, SMILES)`), not the raw
`output/` files. Then pass the CSVs and dict sources explicitly — list `builtin` first so the
combined set resolves both original and paper-scraper names:
```bash
pixi run integrate_paper_scraper \
    --pdcc-datasets DATASETS/PDCC/polymer_drug_concentration_capacity.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_opus_without_conflicts.csv \
                    DATASETS/PDCC/paper_scraper/pdcc_deepseek_without_conflicts.csv \
    --psmiles-dicts builtin DATASETS/PDCC/paper_scraper/paper_scraper_complete_psmiles.json \
    --smiles-dicts  builtin DATASETS/PDCC/paper_scraper/paper_scraper_complete_smiles.json \
    --train         # also split + StandardScaler-scale
```
Run `pixi run integrate_paper_scraper --help` for all flags. It reports rows in / resolved /
featurized, `num_features`, and any names that failed to resolve. Rows with unresolvable names or
malformed numeric fields (e.g. a `WATER_PH` range string like `"6.6-6.8"`) are discarded.

# Thesis experiments (dataset & model comparison)
`bio.run_all_integrated_paper_scraper_experiments` trains the CAPACITY MLP across datasets (original
PDCC, paper-scraper per-model splits, and combined sets) and model sizes, using **leakage-safe
grouped cross-validation** (each polymer–drug curve is held out as a whole — unlike the older
LOOCV-over-augmented runs, whose scores are optimistic). Results and a Q2 leaderboard go to
`RESULTS/thesis_experiments/`, leaving the old `RESULTS/mlp_experiments/` untouched.
```bash
pixi run -e cuda run_integrated_paper_scraper_experiments                       # default sweep
pixi run -e cuda run_integrated_paper_scraper_experiments --datasets original opus deepseek
```
Server-ready scripts (bootstrap → smoke test → build → run → rank → archive) live in `unisi_scripts/`.
