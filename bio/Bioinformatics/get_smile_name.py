from bio.__global__ import SMILES_DICT
from typing import Optional

REVERSE_SMILES_DICT = dict()
smiles_dict_lower = {str(k).lower(): v for k, v in SMILES_DICT.items()}
for name, smile_str in smiles_dict_lower.items():
    if not smile_str: continue
    if smile_str in REVERSE_SMILES_DICT: continue
    REVERSE_SMILES_DICT[smile_str] = name

def get_smile_name(smile_str: str) -> str:
    return REVERSE_SMILES_DICT.get(smile_str, "Unknown")


import pytest
@pytest.mark.parametrize("smile, expected", [
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "aspirin"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "ibuprofen"),
    ("AAAA", "Unknown"),
])
def test_(smile, expected):
    result = get_smile_name(smile)
    assert result == expected, f"Expected {expected}, got {result}"
