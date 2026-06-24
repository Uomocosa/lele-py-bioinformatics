"""
run_all_integrated_paper_scraper_experiments — thesis comparison of CAPACITY models
across datasets (original PDCC + paper-scraper splits, and combined sets) and model
sizes, using leakage-safe grouped cross-validation (group = polymer-drug pair).

This is SEPARATE from `bio/mlp_experiment/run_all_experiments.py` (which is left
untouched) so the existing LOOCV results in RESULTS/mlp_experiments/ are never
overwritten. Output goes to RESULTS/thesis_experiments/.

Invocation (the repo's lazy import system means `python -m` does NOT work):
    pixi run -e cuda run_integrated_paper_scraper_experiments
    pixi run -e cuda run_integrated_paper_scraper_experiments --datasets original opus deepseek
    pixi run python -c "from bio.run_all_integrated_paper_scraper_experiments import rank; rank()"
"""
import yaml
import tyro
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import bio
from bio.__global__ import PDCC_DATASET, RESULTS_DIR
from bio.integrate_paper_scraper import PAPER_SCRAPER_DIR, load_and_merge_datasets
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config
DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
ModelConfig = bio.mlp_experiment.Config.ModelConfig

SAVE_DIR = RESULTS_DIR / "thesis_experiments"

# paper-scraper dict sources (merged on top of the builtin global dicts).
_PS_PSMILES = str(PAPER_SCRAPER_DIR / "paper_scraper_complete_psmiles.json")
_PS_SMILES = str(PAPER_SCRAPER_DIR / "paper_scraper_complete_smiles.json")
_PS_DICTS = dict(psmiles=["builtin", _PS_PSMILES], smiles=["builtin", _PS_SMILES])
_BUILTIN_DICTS = dict(psmiles=["builtin"], smiles=["builtin"])


def _ps(name: str) -> Path:
    return PAPER_SCRAPER_DIR / name


# name -> {csvs: [paths], psmiles: [...], smiles: [...]}. Multi-csv entries are merged
# into one NAME-based CSV before training.
DATASET_REGISTRY = {
    "original": dict(csvs=[PDCC_DATASET], **_BUILTIN_DICTS),
    "opus":     dict(csvs=[_ps("pdcc_opus_without_conflicts.csv")], **_PS_DICTS),
    "deepseek": dict(csvs=[_ps("pdcc_deepseek_without_conflicts.csv")], **_PS_DICTS),
    "kimi":     dict(csvs=[_ps("pdcc_kimi_without_conflicts.csv")], **_PS_DICTS),
    "gemma":    dict(csvs=[_ps("pdcc_gemma4_image_without_conflicts.csv")], **_PS_DICTS),
    "pool":     dict(csvs=[_ps("pdcc_opus_without_conflicts.csv"),
                           _ps("pdcc_deepseek_without_conflicts.csv"),
                           _ps("pdcc_kimi_without_conflicts.csv"),
                           _ps("pdcc_gemma4_image_without_conflicts.csv")], **_PS_DICTS),
    "old_plus_new": dict(csvs=[PDCC_DATASET,
                               _ps("pdcc_opus_without_conflicts.csv"),
                               _ps("pdcc_deepseek_without_conflicts.csv"),
                               _ps("pdcc_kimi_without_conflicts.csv"),
                               _ps("pdcc_gemma4_image_without_conflicts.csv")], **_PS_DICTS),
}

# Baseline (current best) plus the two "slightly bigger" nets, evaluated under the
# SAME grouped-CV methodology so the comparison is fair.
ARCHITECTURES = {
    "hd_16_8_4_4_4":   [16, 8, 4, 4, 4],
    "hd_32_16_8_4_4":  [32, 16, 8, 4, 4],
    "hd_64_32_16_8_4": [64, 32, 16, 8, 4],
}


@dataclass
class Config:
    datasets: List[str] = field(default_factory=lambda: ["original", "opus", "deepseek", "kimi", "gemma"])
    """Which DATASET_REGISTRY entries to run."""
    architectures: List[str] = field(default_factory=lambda: list(ARCHITECTURES))
    """Which ARCHITECTURES entries to run."""
    k_fold: int = 5
    """Grouped folds (-1 = leave-one-group-out; auto-capped to #groups)."""
    seed: int = 42
    max_size: Optional[int] = None
    """Cap rows per dataset (smoke tests only)."""
    save_dir: Path = SAVE_DIR


def _materialize_dataset(name: str, spec: dict, build_dir: Path) -> Path:
    """Return a single NAME-based CSV for a dataset spec, merging multiple CSVs if needed."""
    csvs = [Path(c) for c in spec["csvs"]]
    if len(csvs) == 1:
        return csvs[0]
    merged, _ = load_and_merge_datasets(csvs, deduplicate=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / f"{name}.csv"
    merged.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Built combined dataset '{name}' ({len(merged)} rows) -> {out}")
    return out


def run_with_config(config: Config):
    bio.setup_loguru()
    config.save_dir.mkdir(parents=True, exist_ok=True)
    build_dir = config.save_dir / "_data"

    for ds_name in config.datasets:
        if ds_name not in DATASET_REGISTRY:
            logger.error(f"Unknown dataset '{ds_name}'. Known: {list(DATASET_REGISTRY)}. Skipping.")
            continue
        spec = DATASET_REGISTRY[ds_name]
        csv_file = _materialize_dataset(ds_name, spec, build_dir)

        for arch_name in config.architectures:
            if arch_name not in ARCHITECTURES:
                logger.error(f"Unknown architecture '{arch_name}'. Known: {list(ARCHITECTURES)}. Skipping.")
                continue
            exp = ExperimentConfig(
                name=f"thesis_{ds_name}_{arch_name}",
                k_fold=config.k_fold,
                cv_method="group",
                save_dir=config.save_dir,
                seed=config.seed,
                dataset_config=DatasetConfig(
                    csv_file=csv_file,
                    psmiles_dicts=spec["psmiles"],
                    smiles_dicts=spec["smiles"],
                    max_size=config.max_size,
                    seed=config.seed,
                ),
                model_config=ModelConfig(hidden_dims=ARCHITECTURES[arch_name]),
            )
            logger.info(f"=== Running {exp.name} (grouped {config.k_fold}-fold) ===")
            bio.mlp_experiment.run_with_config(exp)

    rank(config.save_dir)


def rank(base_dir: Path = SAVE_DIR):
    """Aggregate every experiment's fold_predictions.jsonl into a Q2/MAE/RMSE leaderboard.
    Unlike the old LOOCV-only ranker, this ranks any experiment that has predictions and
    records the cv method."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.error(f"Base directory '{base_dir}' does not exist.")
        return

    results = []
    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        predictions_file = exp_dir / "fold_predictions.jsonl"
        if not predictions_file.exists():
            continue
        cv_method = "?"
        config_file = exp_dir / "exp_config.yaml"
        if config_file.exists():
            try:
                cfg = yaml.safe_load(config_file.read_text())
                cv_method = cfg.get("cv_method", "?")
            except yaml.YAMLError:
                pass
        try:
            df_preds = pd.read_json(predictions_file, lines=True)
        except ValueError:
            logger.error(f"Failed to parse predictions for '{exp_dir.name}'.")
            continue
        if df_preds.empty:
            continue
        y_true = np.array(df_preds["actual"].tolist())
        y_pred = np.array(df_preds["predicted"].tolist())
        results.append({
            "Experiment": exp_dir.name,
            "CV": cv_method,
            "Q2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "N": len(y_true),
        })

    df_results = pd.DataFrame(results)
    if df_results.empty:
        logger.info("No experiment results found to rank.")
        return
    df_results = df_results.sort_values(by="Q2", ascending=False).reset_index(drop=True)

    base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / "q2_leaderboard.csv"
    md_path = base_dir / "q2_leaderboard.md"
    df_results.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Thesis Experiment Leaderboard (grouped CV)\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        f.write(df_results.to_markdown(index=False))
    logger.info(f"Leaderboard saved to:\n - {csv_path}\n - {md_path}")
    print(df_results.to_markdown())
    return df_results


def main():
    config = tyro.cli(Config)
    run_with_config(config)


import pytest
@pytest.mark.above10s
def test_smoke():
    # pixi run pytest -rFP -q -s bio/run_all_integrated_paper_scraper_experiments.py::test_smoke -o "addopts="
    config = Config(
        datasets=["original", "opus", "deepseek"],
        architectures=["hd_16_8_4_4_4"],
        k_fold=3,
        max_size=30,
        save_dir=SAVE_DIR / "_smoke_test",
    )
    run_with_config(config)
    lb = rank(config.save_dir)
    assert lb is not None and len(lb) >= 1
