"""
integrate_paper_scraper — load paper-scraper's PDCC datasets, resolve chemical
names to structures using paper-scraper's name->structure dictionaries, run the
existing `bio` featurization pipeline, and emit a featurized, train-ready dataset
(optionally split + scaled).

This is a parameterized, list-friendly version of `bio.Dataset.PDCC.test_usage`.
It is intentionally non-invasive: name resolution is injected via
`PDCCMethod.convert_names_to_smiles.Options` rather than editing the global
PSMILES_DICT / SMILES_DICT.

Invocation note
---------------
This repo uses a custom lazy-import system, so `python -m bio.integrate_paper_scraper`
does NOT work (runpy cannot fetch the module code). Invoke via `-c` (the same pattern
every other entry point in pixi.toml uses), or the `integrate_paper_scraper` pixi task:

    pixi run integrate_paper_scraper -- <args>
    pixi run python -c "from bio.integrate_paper_scraper import main; main()" <args>

Example
-------
Combined conflict-free training set (pass ALL the per-model pool files together;
the global dedup in paper-scraper's `output_filtered/` guarantees no repeated
(P-SMILES, SMILES) tuple across them). With the files copied into the standard dir
you can omit every flag and let the defaults pick up the pool:

    pixi run integrate_paper_scraper

    # ...which is equivalent to spelling it out:
    pixi run python -c "from bio.integrate_paper_scraper import main; main()" \
        --pdcc-datasets DATASETS/PDCC/paper_scraper/pdcc_opus_without_conflicts.csv \
                        DATASETS/PDCC/paper_scraper/pdcc_deepseek_without_conflicts.csv \
                        DATASETS/PDCC/paper_scraper/pdcc_kimi_without_conflicts.csv \
                        DATASETS/PDCC/paper_scraper/pdcc_gemma4_image_without_conflicts.csv \
        --psmiles-dicts DATASETS/PDCC/paper_scraper/paper_scraper_complete_psmiles.json \
        --smiles-dicts  DATASETS/PDCC/paper_scraper/paper_scraper_complete_smiles.json
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

import pandas as pd
import tyro
from loguru import logger

import bio
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import DATASETS_DIR, PDCC_DATASET

# Standard location the user copies paper-scraper's files into.
PAPER_SCRAPER_DIR = DATASETS_DIR / "PDCC" / "paper_scraper"

# Columns of the PDCC data contract. SOURCE is metadata only (not a feature/label).
DEDUP_KEYS = ["POLYMER_USED", "DRUG", "WATER_PH", "CONCENTRATION", "CAPACITY"]

# Dictionary values that mean "no valid structure" and should be treated as missing.
_INVALID_STRUCTURE_VALUES = {"", "nan", "none", "null", "not_a_valid_polymer"}

# In a dict-source list, this token resolves to the in-code global PSMILES_DICT /
# SMILES_DICT (the dicts used for the original PDCC). Anything else is a JSON path.
BUILTIN_TOKEN = "builtin"


@dataclass
class Config:
    pdcc_datasets: List[Path] = field(default_factory=list)
    """One or more PDCC CSVs (6-column schema). Concatenated. Defaults to the original
    PDCC dataset (DATASETS/PDCC/polymer_drug_concentration_capacity.csv) if omitted."""

    psmiles_dicts: List[str] = field(default_factory=lambda: [BUILTIN_TOKEN])
    """Polymer name -> P-SMILES dict sources, merged in order (later wins). Each is the
    'builtin' token (the global PSMILES_DICT) or a JSON file path. Passing values
    REPLACES the default; include 'builtin' to keep the global dict, e.g.
    --psmiles-dicts builtin DATASETS/PDCC/paper_scraper/paper_scraper_complete_psmiles.json"""

    smiles_dicts: List[str] = field(default_factory=lambda: [BUILTIN_TOKEN])
    """Drug name -> SMILES dict sources, merged in order (later wins). 'builtin'
    (the global SMILES_DICT) or JSON file paths. Passing values REPLACES the default."""

    output_csv: Path = PAPER_SCRAPER_DIR / "featurized.csv"
    """Where to write the featurized dataset."""

    deduplicate: bool = True
    """Drop rows that are identical on (POLYMER_USED, DRUG, WATER_PH, CONCENTRATION,
    CAPACITY). This is a NAME-level dedup; cross-paper structure-level conflicts are
    only fully removed by using paper-scraper's `*_without_conflicts.csv` files."""

    train: bool = False
    """If set, also split + StandardScaler-scale the featurized torch dataset."""

    train_validation_test_percentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 42
    max_size: Optional[int] = None
    """Cap the dataset size (rows) before featurization — useful for smoke tests."""


def _is_valid_structure(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in _INVALID_STRUCTURE_VALUES


def resolve_dict_sources(sources: List[str], kind: str) -> Dict[str, str]:
    """Merge name->structure dict sources in order (later wins). Each source is either
    the literal `builtin` token (the in-code global PSMILES_DICT / SMILES_DICT) or a
    path to a `{name: structure}` JSON file. `kind` is 'psmiles' or 'smiles'.
    Empty / invalid structure values are dropped."""
    from bio.__global__ import PSMILES_DICT, SMILES_DICT
    builtin = PSMILES_DICT if kind == "psmiles" else SMILES_DICT

    merged: Dict[str, str] = {}
    for src in sources:
        if str(src) == BUILTIN_TOKEN:
            kept = {k: v for k, v in builtin.items() if _is_valid_structure(v)}
            logger.info(f"Loaded {len(kept):>4} valid {kind} entries from builtin global dict")
        else:
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(
                    f"{kind} dict source not found: {path} (use a JSON file path or '{BUILTIN_TOKEN}')"
                )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            kept = {k: v for k, v in data.items() if _is_valid_structure(v)}
            logger.info(f"Loaded {len(kept):>4} valid {kind} entries from {path.name} "
                        f"({len(data) - len(kept)} empty/invalid dropped)")
        merged.update(kept)  # later source wins on key collision
    return merged


def load_and_merge_datasets(paths: List[Path], deduplicate: bool) -> Tuple[pd.DataFrame, int]:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"PDCC dataset not found: {path}\n"
                f"Pass existing CSV path(s) to --pdcc-datasets. paper-scraper files belong in "
                f"{PAPER_SCRAPER_DIR}."
            )
        df = pd.read_csv(path, encoding="utf-8")
        if df.empty:
            logger.warning(f"Skipping empty dataset: {path.name}")
            continue
        logger.info(f"Loaded {len(df):>4} rows from {path.name}")
        frames.append(df)

    if not frames:
        raise ValueError("No non-empty PDCC datasets were loaded.")

    merged = pd.concat(frames, ignore_index=True)
    if deduplicate:
        keys = [k for k in DEDUP_KEYS if k in merged.columns]
        before = len(merged)
        merged = merged.drop_duplicates(subset=keys).reset_index(drop=True)
        dropped = before - len(merged)
        if dropped:
            logger.info(f"Deduplicated {dropped} identical row(s) on {keys}.")
    return merged, len(frames)


def load_and_merge_dicts(paths: List[Path]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Name->structure dict not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        kept = {k: v for k, v in data.items() if _is_valid_structure(v)}
        logger.info(f"Loaded {len(kept):>4} valid entries from {path.name} "
                    f"({len(data) - len(kept)} empty/invalid dropped)")
        merged.update(kept)  # later file wins on key collision
    return merged


def find_unresolved_names(
    df: pd.DataFrame, psmiles_dict: Dict[str, str], smiles_dict: Dict[str, str]
) -> Tuple[Set[str], Set[str]]:
    """Names present in the data but not resolvable to a valid structure."""
    from bio.Dataset.PDCCMethod.convert_names_to_smiles import is_missing_or_empty

    psmiles_lower = {str(k).lower(): v for k, v in psmiles_dict.items()}
    smiles_lower = {str(k).lower(): v for k, v in smiles_dict.items()}

    poly = df["POLYMER_USED"].astype(str).str.lower()
    drug = df["DRUG"].astype(str).str.lower()
    missing_polymers = {p for p in poly if is_missing_or_empty(p, psmiles_lower)}
    missing_drugs = {d for d in drug if is_missing_or_empty(d, smiles_lower)}
    return missing_polymers, missing_drugs


def run(config: Config):
    bio.setup_loguru()

    # Default dataset = the original PDCC; default dicts = the builtin global dicts.
    pdcc_datasets = config.pdcc_datasets or [PDCC_DATASET]

    # 1. Load + merge data and dicts.
    merged_df, n_contributing = load_and_merge_datasets(pdcc_datasets, config.deduplicate)
    psmiles_dict = resolve_dict_sources(config.psmiles_dicts, "psmiles")
    smiles_dict = resolve_dict_sources(config.smiles_dicts, "smiles")
    rows_in = len(merged_df)

    # 2. Report which names cannot be resolved (so the user knows what to add).
    missing_polymers, missing_drugs = find_unresolved_names(merged_df, psmiles_dict, smiles_dict)

    # 3. Persist the combined frame so PDCC.Config (which takes a CSV path) can read it.
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_csv = config.output_csv.parent / "_combined.csv"
    merged_df.to_csv(combined_csv, index=False, encoding="utf-8")

    # 4. Run the real bio pipeline: resolve names -> structures, then featurize.
    pdcc_config = PDCC.Config(
        csv_file=combined_csv,
        train_validation_test_pecentages=config.train_validation_test_percentages,
        max_size=config.max_size,
        seed=config.seed,
    )
    dataset = PDCC.PDCC(pdcc_config)
    dataset.convert_names_to_smiles(
        PDCCMethod.convert_names_to_smiles.Options(
            psmiles_dict=psmiles_dict,
            smiles_dict=smiles_dict,
        )
    )
    rows_resolved = len(dataset.df)

    torch_dataset = dataset.to_torch_dataset()
    rows_featurized = len(torch_dataset)
    num_features = torch_dataset.num_features

    # 5. Save the featurized frame.
    featurized_df = torch_dataset.df
    featurized_df.to_csv(config.output_csv, index=False, encoding="utf-8")

    # 6. Report.
    logger.info("=" * 60)
    logger.info("paper-scraper integration summary")
    logger.info(f"  datasets merged ......... {n_contributing} of {len(pdcc_datasets)} file(s) (empty skipped)")
    logger.info(f"  rows in ................. {rows_in}")
    logger.info(f"  rows after resolution ... {rows_resolved}")
    logger.info(f"  rows after featurization  {rows_featurized}")
    logger.info(f"  num_features ............ {num_features}")
    logger.info(f"  featurized csv .......... {config.output_csv}")
    if missing_polymers:
        logger.warning(f"  {len(missing_polymers)} unresolved polymer name(s): {sorted(missing_polymers)}")
    if missing_drugs:
        logger.warning(f"  {len(missing_drugs)} unresolved drug name(s): {sorted(missing_drugs)}")
    logger.info("=" * 60)

    if config.train:
        from sklearn.preprocessing import StandardScaler

        trn, val, tst = config.train_validation_test_percentages
        splitted = bio.Dataset.split_dataset(
            dataset=torch_dataset,
            train_percentage=trn,
            validation_percentage=val,
            test_percentage=tst,
            seed=config.seed,
        )
        splitted.scale(
            feature_col_indexes=list(range(num_features)),
            scaler_fn=StandardScaler(),
        )
        logger.info(
            f"  split -> train={len(splitted.train)} "
            f"val={len(splitted.validation)} test={len(splitted.test)} (scaled)"
        )
        return torch_dataset, splitted

    return torch_dataset, None


def main():
    config = tyro.cli(Config)
    run(config)


if __name__ == "__main__":
    main()
