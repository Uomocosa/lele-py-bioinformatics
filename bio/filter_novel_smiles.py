import pandas as pd
from bio.__global__ import RESULTS_DIR
from bio.Dataset.__global__ import ZINC_BASE_CSV, PI1M, COMBINED_PI1M_ZINC
from loguru import logger

def test_smiles():
    dir = RESULTS_DIR / 'smiles_generator' / 'generate_mnt128_t100000000' / '2026_04_02_133739_018144'
    generated_csv_path = dir / 'valid_smiles.csv'
    output_csv_path = generated_csv_path.parent / 'novel_and_valid_smiles.csv'
    filter_novel_smiles(
        generated_csv_path = generated_csv_path, 
        output_csv_path = output_csv_path,
        df_training_data = pd.read_csv(ZINC_BASE_CSV, header=None, names=['smiles']),
    )

def test_psmiles():
    dir = RESULTS_DIR / 'pee_smiles_generator' / 'generate_mnt128_t100000000' / '2026_04_02_162307_599048'
    generated_csv_path = dir / 'valid_smiles.csv'
    output_csv_path = generated_csv_path.parent / 'novel_and_valid_smiles.csv'
    training_csv_path = PI1M
    filter_novel_smiles(
        generated_csv_path = generated_csv_path, 
        output_csv_path = output_csv_path,
        df_training_data = pd.read_csv(PI1M, header=0, names=['smiles']),
    )
    
def test_combined():
    dir = RESULTS_DIR / 'smiles_and_psmiles_generator' / 'generate_mnt128_t100000000' / '2026_04_02_172641_275118'
    generated_csv_path = dir / 'valid_smiles.csv'
    output_csv_path = generated_csv_path.parent / 'novel_and_valid_smiles.csv'
    filter_novel_smiles(
        generated_csv_path = generated_csv_path, 
        output_csv_path = output_csv_path,
        df_training_data = pd.read_csv(COMBINED_PI1M_ZINC, header=None, names=['smiles']),
    )


def filter_novel_smiles(
    generated_csv_path, 
    output_csv_path,
    df_training_data=None, 
):
    """
    Filters out non-unique SMILES both internally (duplicates in generation) 
    and externally (memorized from the training set).
    """
    # 1. Read the generated smiles (Assuming no header column to catch all data)
    df_gen = pd.read_csv(generated_csv_path, header=None, names=['smiles'])
    initial_count = len(df_gen)
    
    # 2. Filter out duplicates from the generated set itself
    df_unique_gen = df_gen.drop_duplicates(subset=['smiles'])
    internal_duplicates = initial_count - len(df_unique_gen)
    logger.info(f"Removed {internal_duplicates} internal duplicates from the generated set.")
    
    # 3. Filter out smiles that were in the training set
    if df_training_data is None:
        df_novel = df_unique_gen
        logger.info("No training set provided. Skipping novelty check.")
    else:
        train_smiles_set = set(df_training_data['smiles'].astype(str))
        df_novel = df_unique_gen[~df_unique_gen['smiles'].isin(train_smiles_set)]
        memorized_count = len(df_unique_gen) - len(df_novel)
        logger.info(f"Removed {memorized_count} SMILES that were already present in the training set.")
    
    logger.info(f"Total fully novel & unique SMILES remaining: {len(df_novel)}")
    df_novel.to_csv(output_csv_path, index=False, header=False)
    logger.info(f"Saved to {output_csv_path}")
    
    return df_novel
