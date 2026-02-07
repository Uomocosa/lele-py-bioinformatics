import lele
from bio.__global__ import DATASETS_DIR, VOCABULARIES_DIR

THIS_FOLDER = lele.P(__file__).parent
HELPER_DIR = THIS_FOLDER / "__HELPER_DIR__"

SMILES_VOCABULARY = VOCABULARIES_DIR / 'SMILES_vocabulary.txt'
ZINC_BASE_CSV = DATASETS_DIR / 'ZINC_base/smiles_copy.csv'
ZINC_ZSCORE_CSV = DATASETS_DIR / 'ZINC_base/smiles_preprocessed_scale-zscore.csv'
PI1M = DATASETS_DIR / 'PI1M/PI1M.csv'
COMBINED_PI1M_ZINC = DATASETS_DIR / 'PI1M+ZINC_base/combined.csv'

assert HELPER_DIR.exists()
assert SMILES_VOCABULARY.exists()
assert ZINC_BASE_CSV.exists()
assert ZINC_ZSCORE_CSV.exists()
assert PI1M.exists()
# assert COMBINED_PI1M_ZINC.exists() # remember to not assert this! It is a generated dataset

def test_():
    pass
