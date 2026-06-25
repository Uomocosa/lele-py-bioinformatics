"""
run_fixed_test_experiments — fair augmentation comparison with a fixed held-out test set.

Design
------
1. The original PDCC is split ONCE by polymer–drug groups per seed (80 % train, 20 % test)
   and saved to DATASETS/PDCC/paper_scraper/:
       original_train_groups_s{seed}.csv  ← used in every experiment for this seed
       original_test_groups_s{seed}.csv   ← never touched during training

2. Before merging, any LLM row whose (POLYMER_USED, DRUG) resolves to the same SMILES-pair
   as a test group is removed to prevent leakage.

3. Each experiment trains on (original_train + optionally filtered LLM data) and is
   evaluated on the same fixed test set, enabling fair apples-to-apples comparison.

4. Running multiple seeds gives a distribution (mean±std Q2) rather than a single noisy
   number — important because a 14-group test split is high variance on its own.

Phase 1  : baseline  vs  baseline + each individual LLM source
Phase 2+ : combinations, decided after seeing Phase 1 results.

Invocation:
    pixi run -e cuda python -c 'from bio.run_fixed_test_experiments import main; main()'
    pixi run -e cuda python -c 'from bio.run_fixed_test_experiments import main; main()' --seeds 42 123 7
"""
import types
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from loguru import logger
import tyro

import bio
from bio.__global__ import PDCC_DATASET, RESULTS_DIR
from bio.integrate_paper_scraper import PAPER_SCRAPER_DIR, resolve_dict_sources, DEDUP_KEYS
from bio.Dataset import PDCC, PDCCMethod
from bio.ML import MLP, MLPMethod
from bio.mlp_experiment.Config import forward_softplus_fn, FeaturizerOptions, IncrementDatasetOptions


SAVE_DIR = RESULTS_DIR / "fixed_test_experiments"
SPLIT_DIR = PAPER_SCRAPER_DIR

_PS_PSMILES = str(PAPER_SCRAPER_DIR / "paper_scraper_complete_psmiles.json")
_PS_SMILES  = str(PAPER_SCRAPER_DIR / "paper_scraper_complete_smiles.json")
# Single combined dict used for ALL name resolution across all experiments so
# the fixed test set is identical (same resolved rows) for every model.
_ALL_DICTS = dict(psmiles=["builtin", _PS_PSMILES], smiles=["builtin", _PS_SMILES])

def _ps(name: str) -> Path:
    return PAPER_SCRAPER_DIR / name


# ─── Registries ───────────────────────────────────────────────────────────────
# Each entry: dict(llm_csvs=[...])
# llm_csvs: zero or more ADDITIONAL CSVs to merge on top of original_train.
# All experiments use the same combined name-resolution dict (_ALL_DICTS) so that
# the test set is featurized identically for every model.

PHASE1_REGISTRY = {
    "baseline":      dict(llm_csvs=[]),
    "plus_opus":     dict(llm_csvs=[_ps("pdcc_opus_without_conflicts.csv")]),
    "plus_deepseek": dict(llm_csvs=[_ps("pdcc_deepseek_without_conflicts.csv")]),
    "plus_kimi":     dict(llm_csvs=[_ps("pdcc_kimi_without_conflicts.csv")]),
    "plus_gemma":    dict(llm_csvs=[_ps("pdcc_gemma4_image_without_conflicts.csv")]),
}

PHASE2_REGISTRY: dict = {}   # populated after seeing Phase 1 results

ARCHITECTURES = {
    "hd_16_8_4_4_4":   [16, 8, 4, 4, 4],
    "hd_32_16_8_4_4":  [32, 16, 8, 4, 4],
    "hd_64_32_16_8_4": [64, 32, 16, 8, 4],
}


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    phase: int = 1
    """1 = PHASE1_REGISTRY (one LLM at a time), 2 = PHASE2_REGISTRY (combinations)."""
    architectures: List[str] = field(default_factory=lambda: list(ARCHITECTURES))
    seeds: List[int] = field(default_factory=lambda: [42, 123, 7])
    """Seeds for the train/test group split. Run multiple seeds for robust comparison.
    Example: --seeds 42 123 7"""
    test_fraction: float = 0.2
    """Fraction of original PDCC groups held out as fixed test set per seed."""
    save_dir: Path = SAVE_DIR
    max_size: Optional[int] = None
    """Cap dataset rows (smoke tests only)."""


# ─── Split ────────────────────────────────────────────────────────────────────

def split_original_pdcc(seed: int, test_fraction: float) -> Tuple[Path, Path]:
    """Split original PDCC by polymer-drug groups. Saves per-seed CSVs; skips if present."""
    train_csv = SPLIT_DIR / f"original_train_groups_s{seed}.csv"
    test_csv  = SPLIT_DIR / f"original_test_groups_s{seed}.csv"

    if train_csv.exists() and test_csv.exists():
        logger.info(f"[seed={seed}] Reusing existing split: {train_csv.name} / {test_csv.name}")
        return train_csv, test_csv

    df = pd.read_csv(PDCC_DATASET)
    groups = df["POLYMER_USED"].astype(str) + " || " + df["DRUG"].astype(str)
    unique_groups = groups.unique()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_test = max(1, int(len(shuffled) * test_fraction))
    test_set = set(shuffled[:n_test])

    mask = groups.isin(test_set)
    df_test  = df[mask].reset_index(drop=True)
    df_train = df[~mask].reset_index(drop=True)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    n_train_groups = len(unique_groups) - n_test
    logger.info(
        f"[seed={seed}] Split PDCC: {len(df_train)} train rows ({n_train_groups} groups)"
        f" / {len(df_test)} test rows ({n_test} groups)"
    )
    return train_csv, test_csv


# ─── Leakage filter ───────────────────────────────────────────────────────────

def _resolve_pairs_fast(
    df: pd.DataFrame, psmiles_dict: dict, smiles_dict: dict
) -> List[Tuple]:
    """Dict-lookup for each row's (polymer_smiles, drug_smiles) WITHOUT dropping rows
    or modifying the DataFrame. Returns NaN for unresolved names."""
    psmiles_lower = {str(k).lower(): v for k, v in psmiles_dict.items()}
    smiles_lower  = {str(k).lower(): v for k, v in smiles_dict.items()}
    poly = df["POLYMER_USED"].astype(str).str.lower().map(psmiles_lower)
    drug = df["DRUG"].astype(str).str.lower().map(smiles_lower)
    return list(zip(poly, drug))


def _build_test_keys(test_csv: Path) -> Set[Tuple[str, str]]:
    """Resolved (polymer_smiles, drug_smiles) pairs in the fixed test set.
    Uses the full combined dict — same one used for all experiments — so the
    set of resolvable test groups is consistent across all models."""
    psmiles_dict = resolve_dict_sources(_ALL_DICTS["psmiles"], "psmiles")
    smiles_dict  = resolve_dict_sources(_ALL_DICTS["smiles"],  "smiles")
    test_df = pd.read_csv(test_csv)
    pairs = _resolve_pairs_fast(test_df, psmiles_dict, smiles_dict)
    keys = {(p, d) for p, d in pairs if pd.notna(p) and pd.notna(d)}
    logger.info(f"  Fixed test set: {len(keys)} resolved polymer-drug groups out of {len(test_df)} rows.")
    return keys


def _filter_leakage(llm_df: pd.DataFrame, test_keys: Set[Tuple]) -> pd.DataFrame:
    """Remove LLM rows whose resolved (polymer, drug) SMILES pair matches a test group.
    Uses the full combined dict (same one used for featurization) so resolution is
    consistent with what the pipeline will use. Preserves original row order and index."""
    psmiles_dict = resolve_dict_sources(_ALL_DICTS["psmiles"], "psmiles")
    smiles_dict  = resolve_dict_sources(_ALL_DICTS["smiles"],  "smiles")
    pairs = _resolve_pairs_fast(llm_df, psmiles_dict, smiles_dict)
    keep = [not (pd.notna(p) and pd.notna(d) and (p, d) in test_keys) for p, d in pairs]
    n_removed = len(keep) - sum(keep)
    if n_removed:
        logger.info(f"  Leakage filter: removed {n_removed}/{len(llm_df)} LLM rows matching test groups.")
    return llm_df[keep].reset_index(drop=True)


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def _build_train_csv(
    exp_name: str, llm_csvs: List[Path],
    train_csv: Path, test_keys: Set[Tuple],
    build_dir: Path, seed: int,
) -> Path:
    """Assemble the training CSV: original_train + leakage-filtered LLM data."""
    if not llm_csvs:
        return train_csv  # baseline: original_train only, no merging needed

    original_df = pd.read_csv(train_csv)
    all_frames  = [original_df]

    for csv_path in llm_csvs:
        if not csv_path.exists():
            logger.warning(f"  LLM CSV not found: {csv_path}. Skipping.")
            continue
        llm_df = pd.read_csv(csv_path)
        filtered = _filter_leakage(llm_df, test_keys)
        if len(filtered) == 0:
            logger.warning(f"  {csv_path.name}: all {len(llm_df)} rows removed by leakage filter. Skipping.")
            continue
        all_frames.append(filtered)

    merged = pd.concat(all_frames, ignore_index=True)
    keys = [k for k in DEDUP_KEYS if k in merged.columns]
    merged = merged.drop_duplicates(subset=keys).reset_index(drop=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    out = build_dir / f"{exp_name}_s{seed}_train.csv"
    merged.to_csv(out, index=False, encoding="utf-8")
    logger.info(f"  Built training CSV: {out.name} ({len(merged)} rows from {len(all_frames)} source(s))")
    return out


def _build_torch_dataset(
    csv_file: Path,
    featurizer_options: FeaturizerOptions,
    increment_options: Optional[IncrementDatasetOptions],
    max_size: Optional[int],
    seed: int,
):
    """Load CSV → PDCC → (optional increment) → name resolution → featurize → PDCCtorch."""
    psmiles_dict = resolve_dict_sources(_ALL_DICTS["psmiles"], "psmiles")
    smiles_dict  = resolve_dict_sources(_ALL_DICTS["smiles"],  "smiles")

    dataset = PDCC.PDCC(config=PDCC.Config(csv_file=csv_file, max_size=max_size, seed=seed))
    if increment_options is not None:
        dataset.increment_dataset(options=increment_options)
    dataset.convert_names_to_smiles(
        PDCCMethod.convert_names_to_smiles.Options(
            psmiles_dict=psmiles_dict,
            smiles_dict=smiles_dict,
        )
    )
    dataset.featurize_fn = lambda df: PDCCMethod.featurize(df, options=featurizer_options)
    return dataset.to_torch_dataset()


# ─── Core experiment runner ───────────────────────────────────────────────────

def run_experiment(
    exp_name: str, arch_name: str, llm_csvs: List[Path],
    train_csv: Path, test_csv: Path,
    test_keys: Set[Tuple],
    build_dir: Path, seed: int,
    max_size: Optional[int],
) -> Optional[dict]:
    featurizer_opts = FeaturizerOptions()
    increment_opts  = IncrementDatasetOptions()

    bio.ML.set_seed(seed)
    logger.info(f"--- {exp_name} / {arch_name} (seed={seed}) ---")

    # ── Training CSV: original_train + filtered LLM data ──
    materialized_train_csv = _build_train_csv(
        exp_name, llm_csvs, train_csv, test_keys, build_dir, seed
    )

    # ── Training torch dataset ──
    train_torch = _build_torch_dataset(
        csv_file=materialized_train_csv,
        featurizer_options=featurizer_opts,
        increment_options=increment_opts,
        max_size=max_size,
        seed=seed,
    )
    if len(train_torch) == 0:
        logger.warning(f"  No featurizable training data. Skipping.")
        return None

    # ── Train/val split (80/20 within training data, for early stopping only) ──
    splitted = bio.Dataset.split_dataset(
        dataset=train_torch,
        train_percentage=0.8,
        validation_percentage=0.2,
        test_percentage=0.0,
        seed=seed,
    )

    # Fit scalers on the 80% train portion, apply in-place to all splits.
    # x_scaler: StandardScaler over all input features
    # y_scaler: MinMaxScaler over CAPACITY so model learns to predict in [0, 1]
    x_scaler = splitted.scale(
        feature_col_indexes=range(train_torch.num_features),
        feature_attribute="X",
        scaler_fn=StandardScaler(),
    )
    y_scaler = splitted.scale(
        feature_col_indexes=[0],
        feature_attribute="y",
        scaler_fn=MinMaxScaler(feature_range=(0, 1)),
    )

    # ── Build and train model ──
    model_config = MLP.Config(
        hidden_dims=ARCHITECTURES[arch_name],
        seed=seed,
    )
    model = MLP(
        splitted_dataset=splitted,
        featurize_fn=None,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        config=model_config,
    )
    model.forward = types.MethodType(forward_softplus_fn, model)
    MLPMethod.train_model(model)

    # ── Fixed test set: load WITHOUT incrementing (real data points only) ──
    test_torch = _build_torch_dataset(
        csv_file=test_csv,
        featurizer_options=featurizer_opts,
        increment_options=None,
        max_size=None,
        seed=seed,
    )
    if len(test_torch) == 0:
        logger.warning(f"  No featurizable test data. Skipping.")
        return None

    # Apply the FITTED x_scaler (no re-fitting) to the test features.
    # test_torch.y stays in original capacity units — these are the actuals.
    test_torch.transform(x_scaler)

    # ── Evaluate on the fixed test set ──
    model.eval()
    actuals, preds = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i in range(len(test_torch)):
            x, y = test_torch[i]
            pred_scaled = model(x.unsqueeze(0).to(device)).cpu().item()
            pred = float(y_scaler.inverse_transform([[pred_scaled]])[0][0])
            preds.append(pred)
            actuals.append(float(y.item()))

    q2   = float(r2_score(actuals, preds))
    mae  = float(mean_absolute_error(actuals, preds))
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    n    = len(actuals)
    logger.success(f"  → Q2={q2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  N_test={n}")

    return {
        "Experiment":   exp_name,
        "Architecture": arch_name,
        "Seed":         seed,
        "Q2":           q2,
        "MAE":          mae,
        "RMSE":         rmse,
        "N_test":       n,
    }


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_with_config(config: Config):
    bio.setup_loguru()
    config.save_dir.mkdir(parents=True, exist_ok=True)
    build_dir = config.save_dir / "_data"

    registry = PHASE1_REGISTRY if config.phase == 1 else PHASE2_REGISTRY
    if not registry:
        logger.error(f"Phase {config.phase} registry is empty. Nothing to run.")
        return

    all_results: List[dict] = []

    for seed in config.seeds:
        logger.info(f"======== Seed {seed} ========")
        train_csv, test_csv = split_original_pdcc(seed, config.test_fraction)

        # Build the set of test group (SMILES) keys once per seed — same for all experiments.
        test_keys = _build_test_keys(test_csv)

        for exp_name, spec in registry.items():
            llm_csvs = [Path(c) for c in spec.get("llm_csvs", [])]
            for arch_name in config.architectures:
                if arch_name not in ARCHITECTURES:
                    logger.error(f"Unknown architecture '{arch_name}'. Known: {list(ARCHITECTURES)}. Skipping.")
                    continue
                result = run_experiment(
                    exp_name=exp_name, arch_name=arch_name,
                    llm_csvs=llm_csvs,
                    train_csv=train_csv, test_csv=test_csv,
                    test_keys=test_keys, build_dir=build_dir,
                    seed=seed, max_size=config.max_size,
                )
                if result:
                    all_results.append(result)

    if not all_results:
        logger.warning("No results collected — nothing to aggregate.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(["Experiment", "Architecture", "Seed"]).reset_index(drop=True)

    detail_path = config.save_dir / "all_runs.csv"
    df.to_csv(detail_path, index=False)
    logger.info(f"Per-run results saved to {detail_path}")

    rank(config.save_dir)


def rank(save_dir: Path = SAVE_DIR) -> Optional[pd.DataFrame]:
    """Aggregate across seeds and produce mean±std leaderboard."""
    save_dir = Path(save_dir)
    detail_path = save_dir / "all_runs.csv"
    if not detail_path.exists():
        logger.error(f"No all_runs.csv found in {save_dir}. Run experiments first.")
        return None

    df = pd.read_csv(detail_path)
    agg = (
        df.groupby(["Experiment", "Architecture"])
        .agg(
            Q2_mean=("Q2",   "mean"),
            Q2_std=("Q2",    "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE",  "std"),
            RMSE_mean=("RMSE","mean"),
            RMSE_std=("RMSE", "std"),
            N_seeds=("Seed", "nunique"),
            N_test=("N_test", "first"),
        )
        .reset_index()
        .sort_values("Q2_mean", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = save_dir / "leaderboard.csv"
    md_path  = save_dir / "leaderboard.md"
    agg.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Fixed-Test Augmentation Study — Leaderboard\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write(agg.to_markdown(index=False))
    logger.info(f"Leaderboard:\n  {csv_path}\n  {md_path}")
    print(agg.to_markdown())
    return agg


def main():
    config = tyro.cli(Config)
    run_with_config(config)


import pytest

@pytest.mark.above10s
def test_smoke():
    # pixi run pytest -rFP -q -s bio/run_fixed_test_experiments.py::test_smoke -o "addopts="
    config = Config(
        phase=1,
        architectures=["hd_16_8_4_4_4"],
        seeds=[42],
        test_fraction=0.4,   # larger test fraction = fewer train groups = faster run
        max_size=30,
        save_dir=SAVE_DIR / "_smoke_test",
    )
    run_with_config(config)
    lb = rank(config.save_dir)
    assert lb is not None and len(lb) >= 1
