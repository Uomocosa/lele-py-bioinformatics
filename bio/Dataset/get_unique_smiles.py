import pandas as pd
from loguru import logger

def get_unique_smiles(smiles: pd.DataFrame) -> pd.DataFrame:
    logger.warning("This function is useless, use it as an example. Use '.drop_duplicates'.")
    return smiles.drop_duplicates()


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_dir = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466"
    dataset_csv = dataset_dir / "valid_smiles.csv"
    csv_file = HELPER_DIR / "get_unique_smiles_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = get_unique_smiles(df.head(10000))
    print(df)
    df.to_csv(csv_file, index=False)


import pytest
from bio.__global__ import BIOINFORMATICS_DIR
@pytest.mark.above10s
@pytest.mark.parametrize("dataset_dir", [
    BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466",
    BIOINFORMATICS_DIR / "PSMILES_checkpoints" / "2026_02_07_121136_450914" / "generate_mnt128_t100000000" / "2026_02_10_094417_233255",
    BIOINFORMATICS_DIR / "SMILES_checkpoints" / "2026_02_07_110058_333737" / "generate_mnt128_t100000000" / "2026_02_10_103702_472515",
])
def test_main(dataset_dir):
    dataset_csv = dataset_dir / "valid_smiles.csv"
    csv_file = dataset_dir / "unique_valid_smiles.csv"
    df = pd.read_csv(dataset_csv)
    df = df.drop_duplicates()
    df = df.rename(columns={"valid_smiles": "unique_valid_smiles"})
    print(df)
    df.to_csv(csv_file, index=False)
