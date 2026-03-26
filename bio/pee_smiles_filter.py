import sys
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
import tyro
import yaml
import torch
import pandas as pd
import lele, bio
from lele.Path import P
from lele.String import get_substring 
from bio.Bioinformatics import Smile
from bio.Dataset import PDCCMethod
from bio.__global__ import PSMILES_DICT, SMILES_DICT, RESULTS_DIR
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

SAVE_DIR = RESULTS_DIR / "filtered_synthetic_candidates"

CHECKPOINT_FOLDER = lele.P(r"./PSMILES_checkpoints") 
CHECKPOINT_TEST_FOLDER = lele.P(r"./PSMILES_checkpoints_test") 
TEST_CSV_FILE = lele.P(r"./RESULTS/pee_smiles_generator/generate_mnt128_t100000000/2026_03_26_172635_822854/valid_smiles.csv")
TRAIN_CSV_FILE = lele.P(r"./DATASETS/PI1M/PI1M.csv")

FeaturizerOptions = bio.Dataset.PDCCMethod.featurize.Options

@dataclass
class FilterConfig():
    csv_file: Path = TEST_CSV_FILE
    save_dir: Optional[Path] = SAVE_DIR
    target_molecule_name: str = "None" # Common name of the target molecule
    target_molecule: Optional[str] = None # Expects SMILES format!
    water_ph: Optional[float] = None 
    csv_train_data: Optional[Path] = None
    max_size: Optional[int] = None
    column_name: str = "valid_smiles"
    
    poly_logp_H_min: Optional[float] = 1.5      # High enough to remain a solid, insoluble phase
    poly_logp_H_max: Optional[float] = None     # No strict upper limit needed
    mol_logp_min: Optional[float] = None        # We usually have a target drug, this is useless
    mol_logp_max: Optional[float] = None        # We usually have a target drug, this is useless
    poly_tpsa_min: Optional[float] = 60.0       # Higher TPSA = more polar sites to grab residues
    poly_tpsa_max: Optional[float] = None       # No upper limit
    max_intermolecular_fmo_gap: Optional[float] = 4.0 # Lower is better for strong binding!
    filter_inverse_net_charge: bool = True
    poly_sa_score_H_max: Optional[float] = 4.5  # 1-10 scale. Must be easily synthesizable at scale!
    
    featurizer_options: FeaturizerOptions = field(default_factory=lambda: FeaturizerOptions())
    # LogP -> Higher is better (but how much higher???).
    # TPSA -> Higher is better (but how much higher???).
    # Gruppo funzionale specifici, gruppi aromatici -> Se entrambi cel'hanno bene. (da riveredere)
    
    def parse(self) -> 'Self':
        if not self.target_molecule_name or self.target_molecule_name == "None":
            logger.warning("target_molecule_name is not set or is 'None'.")
            self.target_molecule_name = 'None'
            return self
        smile_str = bio.get_smiles_from_name(self.target_molecule_name)
        if not smile_str and not self.target_molecule:
            logger.error(f"Could not find SMILES for target_molecule_name: {self.target_molecule_name}. Also no existing target_molecule found.")
            return self
        if not smile_str and self.target_molecule:
            logger.info(f"Could not find SMILES for target_molecule_name: {self.target_molecule_name}, using existing target_molecule: {self.target_molecule}")
            return self
        if smile_str and self.target_molecule:
            if smile_str != self.target_molecule:
                logger.info(f"Found SMILES for target_molecule_name: {self.target_molecule_name}. But it is different from the existing target_molecule: {self.target_molecule}, we will use the most common one.")
        self.target_molecule = smile_str
        return self


"""
# FMO (Frontier Molecular Orbital) Theory

Having the HOMO and LUMO energies for both the polymer and the target drug unlocks a very powerful concept in computational chemistry called Frontier Molecular Orbital (FMO) Theory.
Instead of looking at the internal stability of the polymer (its own HOMO-LUMO gap), you can use these four values to predict how strongly the polymer and the drug will bind to each other via electron sharing (like π−π stacking or charge-transfer complexes).
The Science: Intermolecular FMO Gaps
When two molecules interact, one typically acts as an electron "donor" and the other as an electron "acceptor." The interaction is strongest when the energy level of the donor's highest occupied orbital (HOMO) is very close to the acceptor's lowest unoccupied orbital (LUMO).
You have two possible interactions here:
    Polymer donates to Drug: ΔE1​=∣ELUMO, drug​−EHOMO, poly​∣
    Drug donates to Polymer: ΔE2​=∣ELUMO, poly​−EHOMO, drug​∣
The smaller of these two gaps represents the dominant interaction pathway. If this "cross-gap" is small, the polymer and the drug will have a strong electronic affinity for one another—exactly what you want for a wastewater sponge!
"""

"""
# Check for aromatic ring

The Argument: Do we actually need it?
Here is where we debate. You might not actually need this filter, because you already built a trap for aromatic rings without realizing it.
Your FMO (Frontier Molecular Orbital) gap filter is currently doing the heavy lifting. How do molecules achieve small HOMO-LUMO gaps that allow for strong donor-acceptor interactions? By having highly conjugated π-electron systems—which almost exclusively means aromatic rings.
The FMO filter already implicitly screens for aromaticity. It's the reason the surviving polymer from your last test had two massive benzene rings (a styrene group and an azobenzene group). The quantum math naturally selected them.
"""

def main():
    bio.setup_loguru()
    config = tyro.cli(FilterConfig)
    run_with_config_and_save(config)

@dataclass
class SimplerConfig:
    target_molecule_name: str
    max_size: Optional[int] = None

def simple_main():
    bio.setup_loguru()
    cli_args = tyro.cli(SimpleArgs)
    run_for_target_molecule(
        target_molecule_name = cli_args.target_molecule_name,
        max_size = cli_args.max_size,
    )


import pytest
@pytest.mark.above10s
def test_():
    # pixi run pytest -rFP -q -s bio\pee_smiles_filter.py::test_ -o "addopts="
    filter_config = FilterConfig(
        target_molecule_name = "aspirin",
        water_ph = 8.2,
        max_size = 10,
        save_dir = None,
    )
    df = run_with_config(filter_config)
    df = clean_output_df(df)
    print(df)
    
import pytest
@pytest.mark.above10s
def test_aspirin():
    # pixi run pytest -rFP -q -s bio\pee_smiles_filter.py::test_aspirin -o "addopts="
    bio.setup_loguru()
    run_for_target_molecule(
        target_molecule_name = "aspirin", # present in pdcc
        max_size = None,
    )

import pytest
@pytest.mark.above10s
def test_metformin():
    # pixi run pytest -rFP -q -s bio\pee_smiles_filter.py::test_metformin -o "addopts="
    bio.setup_loguru()
    run_for_target_molecule(
        target_molecule_name = "metformin", # present in pdcc
        max_size = None,
    )

import pytest
@pytest.mark.above10s
def test_lisinopril():
    # pixi run pytest -rFP -q -s bio\pee_smiles_filter.py::test_lisinopril -o "addopts="
    bio.setup_loguru()
    run_for_target_molecule(
        target_molecule_name = "lisinopril", # not present in pdcc at the moment
        max_size = None,
    )

import pytest
@pytest.mark.above10s
def test_ibuprofen():
    # pixi run pytest -rFP -q -s bio\pee_smiles_filter.py::test_ibuprofen -o "addopts="
    bio.setup_loguru()
    run_for_target_molecule(
        target_molecule_name = "ibuprofen", # not present in pdcc at the moment
        max_size = None,
    )


def run_with_config_and_save(config: FilterConfig):
    assert config.target_molecule is not None
    filtered_df = run_with_config(config)
    clean_df = clean_output_df(filtered_df)
    if config.save_dir is not None:
        config.save_dir.mkdir(parents=True, exist_ok=True)
        out_csv = config.save_dir / f"target_{config.target_molecule_name}.csv"
        clean_df.to_csv(out_csv, index=False, float_format="%.4f")
        logger.info(f"Saved {len(clean_df)} candidates to {out_csv}")
        
        out_yaml = config.save_dir / f"target_{config.target_molecule_name}_filter_config.yaml"
        yaml.SafeDumper.add_multi_representer(
            Path, 
            lambda dumper, data: dumper.represent_str(str(data))
        )
        formatted_config = yaml.safe_dump(
            asdict(config), 
            default_flow_style=False, 
            sort_keys=False
        )
        with open(out_yaml, "w") as f: f.write(formatted_config)
        logger.info(f"Saved configuration to {out_yaml}")



def run_for_target_molecule(target_molecule_name: str, max_size: Optional[int] = None):
    config = FilterConfig()
    config.csv_train_data = TRAIN_CSV_FILE
    config.max_size = max_size
    config.target_molecule_name = target_molecule_name
    config.target_molecule = bio.get_smiles_from_name(target_molecule_name)
    config.water_ph = 8.2
    config.featurizer_options = FeaturizerOptions(
        molecule_features_to_calculate = [
            'logp', 
            'logd', 
            'homo_lumo_eV', 
            'net_charge', 
            # 'fingerprint',
        ],
        polymer_features_to_calculate = [
            'logp', 
            'logd', 
            'homo_lumo_eV', 
            'net_charge', 
            # 'fingerprint'
        ],
    )
    run_with_config_and_save(config)


def clean_output_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the identifier and filtered property columns."""
    cols_to_keep = [
        'POLYMER_USED',
        'DRUG',
        'poly_logp_H',
        'drug_logp',
        'poly_topological_surface_area_sum_fullpolymerfeaturizer',
        'sa_score_H',
        'poly_homo_eV_H',
        'poly_lumo_eV_H',
        'drug_homo_eV',
        'drug_lumo_eV',
        'poly_charge_at_WATER_PH_H',
        'drug_charge_at_WATER_PH'
    ]
    # Only keep the columns that actually exist to prevent KeyErrors
    final_cols = [c for c in cols_to_keep if c in df.columns]
    return df[final_cols]


    
def run_with_config(config: FilterConfig) -> pd.DataFrame:
    config.parse()
    df = pd.read_csv(config.csv_file)
    if config.max_size: df = df.head(config.max_size)
    
    initial_count = len(df)
    df = df.drop_duplicates(subset=[config.column_name])
    logger.info(f"Dropped {initial_count - len(df)} duplicate molecules from generation.")
    
    if config.csv_train_data and config.csv_train_data.exists():
        train_df = pd.read_csv(config.csv_train_data)
        possible_cols = ['smiles', 'SMILES', 'psmiles', 'PSMILES', 'valid_smiles']
        train_col = next((col for col in possible_cols if col in train_df.columns), train_df.columns[0])
        train_smiles = set(train_df[train_col].dropna().astype(str))
        pre_train_drop_count = len(df)
        df = df[~df[config.column_name].astype(str).isin(train_smiles)]
        logger.info(f"Dropped {pre_train_drop_count - len(df)} molecules already present in the training set.")
    
    df = df.reset_index(drop=True)
    dataset_dir = config.csv_file.parent
    csv_file_unique = dataset_dir / "unique_valid_psmiles.csv"
    df = df.rename(columns={config.column_name: "unique_valid_psmiles"})
    df.to_csv(csv_file_unique, index=False, na_rep='NaN')
    config.column_name = "unique_valid_psmiles"
    
    target_col = "POLYMER_USED" if "POLYMER_USED" in df.columns else config.column_name
    df = bio.Metric.calculate_sa_score(df, column_name=target_col)
        
    df = prepare_dataframe_for_featurizing(df, config)
    identifiers = df[['POLYMER_USED', 'DRUG']].copy()
    df = PDCCMethod.featurize(df, config.featurizer_options)
    if 'POLYMER_USED' not in df.columns:
        df['POLYMER_USED'] = identifiers['POLYMER_USED']
    if 'DRUG' not in df.columns:
        df['DRUG'] = identifiers['DRUG']
    df = apply_filters(df, config)
    return df


def prepare_dataframe_for_featurizing(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Transforms raw generation data into the required format."""
    df_out = df.copy()

    if config.column_name in df_out.columns:
        df_out = df_out.rename(columns={config.column_name: "POLYMER_USED"})
    else:
        raise NameError(f"Column '{config.column_name}' not found in the input CSV.")
    
    if "DRUG" in df_out.columns and config.target_molecule:
        logger.info("Dataframe already has a DRUG column, overwriting with config.target_molecule")
        df_out["DRUG"] = config.target_molecule
    elif "DRUG" not in df_out.columns and config.target_molecule:
        df_out["DRUG"] = config.target_molecule
    elif "DRUG" not in df_out.columns and not config.target_molecule:
        raise ValueError("Expected either:\n\t>>> column 'DRUG' in the input CSV.\n\t>>> config.target_molecule declared.")
    
    if "WATER_PH" in df_out.columns and config.water_ph is not None:
        logger.info("Dataframe already has a WATER_PH column, overwriting with config.water_ph")
        df_out["WATER_PH"] = config.water_ph
    elif "WATER_PH" not in df_out.columns and config.water_ph is not None:
        df_out["WATER_PH"] = config.water_ph
    elif "WATER_PH" not in df_out.columns and config.water_ph is None:
        raise ValueError("Expected either:\n\t>>> column 'WATER_PH' in the input CSV.\n\t>>> config.water_ph declared.")
    
    return df_out


def apply_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Filters the dataframe based on the thresholds in FilterConfig."""
    # Start with a mask where everything is True
    mask = pd.Series(True, index=df.index)
    initial_count = len(df)
    
    TPSA_COL = "poly_topological_surface_area_sum_fullpolymerfeaturizer"
    # Safely collect all columns we actually care about dropping NAs for
    subset_cols = [c for c in [
        'poly_logp_H', 'drug_logp', TPSA_COL, 'sa_score_H', 
        'poly_homo_eV_H', 'poly_lumo_eV_H', 'drug_homo_eV', 'drug_lumo_eV',
        'poly_charge_at_WATER_PH_H', 'drug_charge_at_WATER_PH'
    ] if c in df.columns]
    
    # Filter by LogP
    if config.poly_logp_H_min is not None and 'poly_logp_H' in df.columns:
        mask = mask & (df['poly_logp_H'] >= config.poly_logp_H_min)
    if config.poly_logp_H_max is not None and 'poly_logp_H' in df.columns:
        mask = mask & (df['poly_logp_H'] <= config.poly_logp_H_max)
    if config.mol_logp_min is not None and 'drug_logp' in df.columns:
        mask = mask & (df['drug_logp'] >= config.mol_logp_min)
    if config.mol_logp_max is not None and 'drug_logp' in df.columns:
        mask = mask & (df['drug_logp'] <= config.mol_logp_max)
        
    # Filter by TPSA
    if config.poly_tpsa_min is not None and TPSA_COL in df.columns:
        mask = mask & (df[TPSA_COL] >= config.poly_tpsa_min)
    if config.poly_tpsa_max is not None and TPSA_COL in df.columns:
        mask = mask & (df[TPSA_COL] <= config.poly_tpsa_max)
        
    # Filter by Intermolecular FMO Gap (Donor-Acceptor Interaction)
    if config.max_intermolecular_fmo_gap is not None:
        poly_homo = 'poly_homo_eV_H'
        poly_lumo = 'poly_lumo_eV_H'
        drug_homo = 'drug_homo_eV'
        drug_lumo = 'drug_lumo_eV'
        if all(col in df.columns for col in [poly_homo, poly_lumo, drug_homo, drug_lumo]):
            # Calculate the two possible interaction pathways (absolute difference)
            gap1 = (df[drug_lumo] - df[poly_homo]).abs() # Polymer donates to Drug
            gap2 = (df[poly_lumo] - df[drug_homo]).abs() # Drug donates to Polymer
            # Find the dominant interaction (the smaller gap)
            dominant_gap = pd.concat([gap1, gap2], axis=1).min(axis=1)
            # Keep only molecules where the dominant gap is below our threshold
            mask = mask & (dominant_gap <= config.max_intermolecular_fmo_gap)
        
    # Filter by NET CHARGE
    if config.filter_inverse_net_charge:
        poly_charge_col = 'poly_charge_at_WATER_PH_H'
        drug_charge_col = 'drug_charge_at_WATER_PH'
        if poly_charge_col in df.columns and drug_charge_col in df.columns:
            # If the signs are opposite, their product will be less than 0.
            # E.g., (-1.2 * 0.8) = -0.96 (Passes filter)
            # E.g., (-1.2 * -0.5) = 0.60 (Fails filter)
            mask = mask & ((df[poly_charge_col] * df[drug_charge_col]) < 0)

    # Filter by SA SCORE
    if config.poly_sa_score_H_max is not None and 'sa_score_H' in df.columns:
        mask = mask & (df['sa_score_H'] <= config.poly_sa_score_H_max)

    filtered_df = df[mask].dropna(subset=subset_cols)
    logger.info(f"Filtered out a total of {initial_count - len(filtered_df)} molecules (out of {initial_count})")
    
    if filtered_df.empty: logger.info("No molecules survived the filters.")
    return filtered_df


def reorder_df(df):
    cols = df.columns.tolist()
    front_cols = ['POLYMER_USED', 'DRUG']
    for col in front_cols:
        if col in cols: cols.remove(col)
    df = df[front_cols + cols]
    return df
