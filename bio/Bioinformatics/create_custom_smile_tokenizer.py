from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from loguru import logger
import lele

def create_custom_smile_tokenizer(vocab_path):
    """
    THIS IS USELESS, DO NOT USE THIS. TAKE THIS AS AN EXAMPLE!
    instead use: 
        from deepchem.feat.smiles_tokenizer import SmilesTokenizer
        SmilesTokenizer(vocab_path)
    """
    vocab_path = lele.P(vocab_path)
    assert vocab_path.exists()
    tokenizer = SmilesTokenizer(vocab_path)
    return tokenizer

import pytest
@pytest.mark.skip(reason="THIS IS USELESS, DO NOT USE THIS. TAKE THIS AS AN EXAMPLE!")
def test_():
    import bio
    from bio.Dataset.__global__ import ZINC_BASE_CSV, SMILES_VOCABULARY
    smiles = bio.Dataset.UnlabeledSmiles.load(ZINC_BASE_CSV, max_dataset_size=100)
    print(f"len(smiles): {len(smiles)}")
    tokenizer = create_custom_smile_tokenizer(SMILES_VOCABULARY)
    encodings = [tokenizer.encode(s) for s in smiles]
    check = [tokenizer.decode(e) for e in encodings]
    for s in smiles[:10]: print(s)
    for e in encodings[:10]: print(e)
    for c in check[:10]: print(c)
