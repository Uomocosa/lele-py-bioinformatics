import pandas as pd
from typing import Optional

import bio
from loguru import logger

def add_metrics_to_df(
    df: pd.DataFrame, 
    column_name: str, 
    ph_min: float = 7.0, 
    ph_max: float = 7.4, 
    precision: float = 0.5,
    train_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy()
    df = bio.Metric.calculate_logp(df, column_name)
    df = bio.Metric.calculate_logd(df, column_name, ph_min, ph_max, precision)
    df = bio.Metric.calculate_homo_lumo_gap(df, column_name)
    df = bio.Metric.add_polymetrix_to_df(df, column_name)
    return df
    

import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "add_metrics_to_df_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = add_metrics_to_df(df.head(10), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
