from bio.__global__ import PSMILES_DICT
from typing import Optional

REVERSE_PSMILES_DICT = dict()
for name, psmile_str in PSMILES_DICT.items():
    if not psmile_str: continue
    if psmile_str in REVERSE_PSMILES_DICT: continue
    REVERSE_PSMILES_DICT[psmile_str] = name

def get_psmile_name(psmile_str: str) -> str:
    return REVERSE_PSMILES_DICT.get(psmile_str, "Unknown")


import pytest
@pytest.mark.parametrize("smile, expected", [
    ("*CC*", "polyethylene"),
    ("*CC(C)(C(=O)O)*", "C-mA"),
    ("AAAA", "Unknown"),
])
def test_(smile, expected):
    result = get_psmile_name(smile)
    assert result == expected, f"Expected {expected}, got {result}"
