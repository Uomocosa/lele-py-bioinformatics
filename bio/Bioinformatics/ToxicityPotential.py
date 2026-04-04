"""
TODO: Ask if this makes sense!
NOTE! Made with Gemini (AI)

For toxicity, we CANNOT look at a "polymer toxicity" score. There is none.
Instead, analyze the monomer for structural alerts (PAINS/BRENK).
Using RDKit, as the monomer's reactivity is the primary toxicity risk during 
synthesis and degradation.

Also: For P-SMILES we need to replace star connection points (*) with 
Methyl groups (C) or Hydrogens to make it a valid molecule for RDKit checking
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from enum import IntEnum
from loguru import logger
import bio

class ToxicityPotential(IntEnum):
    UNKNOWN = -1
    LOW = 0
    HIGH = 1
    

def check(smile_or_psmile: str) -> ToxicityPotential:
    # 1. Clean the P-SMILES to get a pseudo-monomer
    # We replace star connection points (*) with Methyl groups (C) or Hydrogens
    # to make it a valid molecule for RDKit checking.
    smile = smile_or_psmile.replace("*", "C") 
    mol = Chem.MolFromSmiles(smile)
    if not mol: return ToxicityPotential.UNKNOWN

    # 2. Initialize Toxicity Filters (e.g., PAINS, BRENK, NIH)
    # These catch reactive groups often associated with toxicity.
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    
    # 3. Check for matches
    if catalog.HasMatch(mol):
        matches = [match.GetDescription() for match in catalog.GetMatches(mol)]
        logger.debug(f"Warning: Potential Toxic/Reactive Alerts found: {matches}")
        return ToxicityPotential.HIGH
    else:
        logger.debug(f"Low risk: No standard structural toxicity alerts found in monomer.")
        return ToxicityPotential.LOW


import pytest
@pytest.mark.parametrize("input,expected", [
    ("*CC(*)(C)C(=O)OCC", ToxicityPotential.LOW) # NOTE! THIS IS IRRITANT! Getting LOW for PMMA (this test string) is correct. While the monomer (methyl methacrylate) is an irritant, it does not contain the heavy-hitting structural alerts (like nitro groups, epoxides, or thiocarbonyls) that these filters typically catch
])
def test_(input, expected):
    result = check(input)
    assert result == expected, f"Got: {result}"
