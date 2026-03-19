"""
OLD!
TO BE REMOVED!
"""

# import tyro
# import torch
# import pandas as pd
# import time, warnings
# from pathlib import Path
# from typing import Optional, Callable
# from dataclasses import dataclass
# import lele, bio
# from lele.Path import P
# from lele.String import get_substring 
# from bio.Bioinformatics import Smile
# from loguru import logger
# import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

# CHECKPOINT_FOLDER = lele.P(r"./PSMILES_checkpoints") 
# CHECKPOINT_TEST_FOLDER = lele.P(r"./PSMILES_checkpoints_test") 
# TEST_CSV_FILE = lele.P(r"./PSMILES_checkpoints/2026_02_07_121136_450914/generate_mnt128_t100000000/2026_02_10_094417_233255/valid_smiles.csv")
# TRAIN_CSV_FILE = lele.P(r"./DATASETS/PI1M/PI1M.csv")

# @dataclass
# class FilterConfig():
#     csv_file: Path = TEST_CSV_FILE
#     csv_train_data: Optional[Path] = None
#     max_size: Optional[int] = None
#     column_name: str = "valid_smiles"
#     ph_min: float = 7.0
#     ph_max: float = 7.4
#     precision: float = 0.5
#     train_data: Optional[pd.DataFrame] = None
#     # LogP -> Meglio alto.
#     # Alto TPSA -> Meglio alto.
#     # Gruppo funzionale specifici, gruppi aromatici -> Se entrambi cel'hanno bene. (da riveredere)
#     # pKa, pH -> da rivedere


# import pytest
# @pytest.mark.above10s
# def main():
#     config = tyro.cli(FilterConfig)
#     run_with_config(config)

# def test_():
#     config = FilterConfig()
#     config.csv_train_data = TRAIN_CSV_FILE
#     config.max_size = 10
#     run_with_config(config)
    
# def run_with_config(config: FilterConfig):
#     dataset_csv = config.csv_file
#     dataset_dir = dataset_csv.parent
#     csv_file_unique = dataset_dir / "unique_valid_psmiles.csv"
#     df = pd.read_csv(dataset_csv)
#     if config.max_size: df = df.head(config.max_size)
#     df = df.drop_duplicates()
#     df = df.rename(columns={config.column_name: "unique_valid_psmiles"})
#     df = add_metrics_to_df(df, "unique_valid_psmiles", config)
#     print(df)
#     df.to_csv(csv_file_unique, index=False, na_rep='NaN')


# def add_metrics_to_df(
#     df: pd.DataFrame, 
#     column_name: str,
#     config: FilterConfig,
# ) -> pd.DataFrame:
#     df = df.copy()
#     df = bio.Metric.calculate_logp(df, column_name)
#     df = bio.Metric.calculate_logd(df, column_name, config.ph_min, config.ph_max, config.precision)
#     df = bio.Metric.calculate_sa_score(df, column_name)
#     df = bio.Metric.calculate_homo_lumo_gap(df, column_name)
#     df = bio.Metric.add_polymetrix_to_df(df, column_name)
#     if not config.train_data: return df
#     df = bio.PolyGen.check_novelty(
#         df_generated = df, 
#         df_t rain = pd.read_csv(config.train_data), 
#         column_name = column_name
#     )
#     return df
