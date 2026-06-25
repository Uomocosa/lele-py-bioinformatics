"""
search_for_best_config — curated grid search for a good CAPACITY-model config under
the leakage-safe grouped cross-validation (group = polymer-drug pair).

Analogous to the old `bio/mlp_experiment/run_all_experiments.py`, but:
  - evaluates with grouped CV (honest) instead of LOOCV-over-augmented (optimistic),
  - runs on a single chosen dataset (default: original PDCC),
  - writes its own isolated leaderboard under RESULTS/config_search/<dataset>/.

IMPORTANT — read before quoting any number:
  This ranks many configs by their grouped-CV Q2 and reports the best. The TOP Q2 is
  OPTIMISTICALLY BIASED: picking the max over many noisy trials is selection bias
  (multiple comparisons). Use this to FIND a promising config, but do NOT report the
  winner's CV Q2 as the model's accuracy without confirming it on data that played no
  part in the search (a held-out group test set or nested CV). The old code is untouched
  and RESULTS/mlp_experiments/ is never overwritten.

Invocation (the repo's lazy import system means `python -m` does NOT work):
    pixi run -e cuda search_for_best_config
    pixi run -e cuda search_for_best_config --dataset old_plus_new
    pixi run python -c "from bio.search_for_best_config import rank; rank()"
"""
import tyro
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field
from typing import List, Optional
import bio
from bio.__global__ import RESULTS_DIR
from bio.run_all_integrated_paper_scraper_experiments import (
    DATASET_REGISTRY, _materialize_dataset, rank,
)
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config
DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
ModelConfig = bio.mlp_experiment.Config.ModelConfig

SAVE_DIR = RESULTS_DIR / "config_search"

# Curated, interpretable axes. x-scaler is fixed to "standard" (clearly beneficial in the
# old sweep); we vary the y-scaler, loss, output activation and architecture.
ARCHITECTURES = {
    "hd_16_8_4_4_4":   [16, 8, 4, 4, 4],   # current best baseline
    "hd_32_16_8_4_4":  [32, 16, 8, 4, 4],
    "hd_64_32_16_8_4": [64, 32, 16, 8, 4],
    "hd_16_16_16":     [16, 16, 16],
    "hd_32_32_32":     [32, 32, 32],
}

# "none" maps to no y-scaler (None); kept as a string so it is CLI-friendly.
_Y_SCALER_NONE = "none"


@dataclass
class Config:
    dataset: str = "original"
    """Which DATASET_REGISTRY entry to search on (original, opus, deepseek, kimi, gemma, pool, old_plus_new)."""
    architectures: List[str] = field(default_factory=lambda: [
        "hd_16_8_4_4_4", "hd_32_16_8_4_4", "hd_64_32_16_8_4",
    ])
    criteria: List[str] = field(default_factory=lambda: ["mse", "mae"])
    forward_fns: List[str] = field(default_factory=lambda: ["softplus", "basic"])
    y_scalers: List[str] = field(default_factory=lambda: ["min_max_range01", _Y_SCALER_NONE])
    """y-scaler per config; use 'none' for no y-scaling."""
    k_fold: int = 5
    """Grouped folds (-1 = leave-one-group-out; auto-capped to #groups)."""
    seed: int = 42
    max_size: Optional[int] = None
    save_dir: Path = SAVE_DIR


def _build_grid(config: Config):
    for arch, crit, fwd, yscaler in product(
        config.architectures, config.criteria, config.forward_fns, config.y_scalers
    ):
        if arch not in ARCHITECTURES:
            logger.error(f"Unknown architecture '{arch}'. Known: {list(ARCHITECTURES)}. Skipping.")
            continue
        yield arch, crit, fwd, yscaler


def run_with_config(config: Config):
    bio.setup_loguru()
    if config.dataset not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{config.dataset}'. Known: {list(DATASET_REGISTRY)}")

    spec = DATASET_REGISTRY[config.dataset]
    save_dir = config.save_dir / config.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_file = _materialize_dataset(config.dataset, spec, save_dir / "_data")

    grid = list(_build_grid(config))
    logger.warning(
        f"Grid search over {len(grid)} configs on '{config.dataset}'. The TOP grouped-CV Q2 "
        f"will be OPTIMISTIC (selection bias) — confirm the winner on held-out groups before quoting it."
    )

    for arch, crit, fwd, yscaler in grid:
        y_scaler_fn = None if yscaler == _Y_SCALER_NONE else yscaler
        name = f"search_{config.dataset}_{arch}_{crit}_{fwd}_y-{yscaler}"
        exp = ExperimentConfig(
            name=name,
            k_fold=config.k_fold,
            cv_method="group",
            save_dir=save_dir,
            seed=config.seed,
            x_scaler_fn="standard",
            y_scaler_fn=y_scaler_fn,
            forward_fn=fwd,
            dataset_config=DatasetConfig(
                csv_file=csv_file,
                psmiles_dicts=spec["psmiles"],
                smiles_dicts=spec["smiles"],
                max_size=config.max_size,
                seed=config.seed,
            ),
            model_config=ModelConfig(
                hidden_dims=ARCHITECTURES[arch],
                criterion_fn=crit,
            ),
        )
        logger.info(f"=== {name} (grouped {config.k_fold}-fold) ===")
        bio.mlp_experiment.run_with_config(exp)

    lb = rank(save_dir)
    logger.warning(
        "Reminder: the leaderboard's top Q2 is optimistic (best-of-many). Treat it as a candidate, "
        "not a reported accuracy."
    )
    return lb


def main():
    config = tyro.cli(Config)
    run_with_config(config)


import pytest
@pytest.mark.above10s
def test_smoke():
    # pixi run pytest -rFP -q -s bio/search_for_best_config.py::test_smoke -o "addopts="
    config = Config(
        dataset="original",
        architectures=["hd_16_8_4_4_4"],
        criteria=["mse"],
        forward_fns=["softplus"],
        y_scalers=["min_max_range01"],
        k_fold=3,
        max_size=30,
        save_dir=SAVE_DIR / "_smoke_test",
    )
    lb = run_with_config(config)
    assert lb is not None and len(lb) >= 1
