"""
IMPORTANT: / TODO: 
PSmile strings like: 
    *CC(CCCCCCCCCC)CC(C*)COCCC
    *CCCC1CN(c2ccc(C#Cc3ccc(C*)cc3)cc2)C1OC(=O)c1ccccc1OC(=O)C(COCCCCOc1ccc(-c2ccccc2)cc1)C(=O)OCCCCCCCCOc1ccc(-c2ccc(-c3ccc(OC)cc3)cc2)cc1
    *=C=C=C(CCC)C(CCCCCCCCCCC*)COCCOCC=CCCOCCOCCOC
    ...
Are NOT recognized as valid PSmile strings.
"""


import re
from loguru import logger
import lele, bio
PSmile = bio.Bioinformatics.PSmile.PSmile


def from_str(psmile: str) -> PSmile:
    if not psmile.strip().startswith("*"): return None
    if psmile.strip().endswith("(*)"): return None
    if "(*)" in psmile and psmile.endswith("(*)"): return None
    
    pattern = re.compile(r"(\(\*\)|\*)")
    matches = list(pattern.finditer(psmile))
    # NOTE! For rdkit "*" is treated as a wildcard
    # We expect exactly 2 wildcards (start and end markers)
    if len(matches) != 2: return None
    
    start_match = matches[0]
    end_match = matches[1]

    # Validate the second wildcard type against its position
    is_simple_star = end_match.group() == "*"
    is_at_end = end_match.end() == len(psmile)

    # Rule derived from tests:
    # 1. If marker is '*', it MUST be at the end. (Valid: *...*, Invalid: *...*Text)
    # 2. If marker is '(*)', it MUST NOT be at the end. (Valid: *...(*)Text, Invalid: *...(*))
    if is_simple_star != is_at_end: return None
    
    start_index = matches[0].end()
    end_index = matches[1].start()
    repeating_unit = psmile[start_index:end_index]
    logger.debug(f"repeating_unit: {repeating_unit}")
    if not repeating_unit: return None
    
    pre_chain = psmile[:start_index]
    pre_chain = pre_chain.removeprefix('*')
    post_chain = psmile[end_index:]
    post_chain = post_chain.removeprefix('(*)').removeprefix('*')
    return PSmile(pre_chain, repeating_unit, post_chain)


import pytest
@pytest.mark.parametrize("input, output", [
    ("*CCCC1CCCC(*)c2ccc2Ncc1", PSmile("", "CCCC1CCCC", "c2ccc2Ncc1")),
    ("*C(=O)O*", PSmile("", "C(=O)O", "")),
    ("*C(=O)O(*)", None), # Expected * at the end
    ("*CCCC1CCCC*c2ccc2Ncc1", None), # Expected (*) in the middle
    ("C1=CC=C", None), # Invalid p-smiles (no stars), but valid molecule
    ("InvalidSMILESString", None)
])
def test_(input, output):
    assert from_str(input) == output
