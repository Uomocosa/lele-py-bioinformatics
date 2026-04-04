"""
TODO: Ask if this makes sense!
NOTE! Made with Gemini (AI)

We will check for three major categories of pollutants:
- PFAS ("Forever Chemicals"): Chains of carbons saturated with Fluorine. 
  These do not degrade.
- Halogenated Aromatics: Often associated with persistent organic 
  pollutants (like PCBs or DDT-like structures).
- High Bioaccumulation Risk: If the standard "LogP" (fat solubility) 
  is too high (> 5), the substance tends to accumulate in fish/wildlife.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from enum import IntEnum
from loguru import logger
import pytest

class PollutionPotential(IntEnum):
    UNKNOWN = -1
    LOW = 0
    HIGH = 1

def check_pollution(smile_or_psmile: str) -> PollutionPotential:
    # 1. Cap the polymer to make a monomer (Same as before)
    smile = smile_or_psmile.replace("*", "C") 
    mol = Chem.MolFromSmiles(smile)
    if not mol: return PollutionPotential.UNKNOWN

    # --- DEFINE CUSTOM POLLUTANT PATTERNS (SMARTS) ---
    
    # A. PFAS (Per- and polyfluoroalkyl substances)
    # Logic: Look for a Carbon attached to at least 2 Fluorines, 
    # appearing in a chain (simplified detection)
    pfas_pattern = Chem.MolFromSmarts("[CX4](F)(F)[CX4](F)(F)") 
    
    # B. Phthalates (Common plasticizer pollutant)
    # Logic: Benzene ring with two ester groups ortho to each other
    phthalate_pattern = Chem.MolFromSmarts("c1ccccc1C(=O)O") 

    # --- CHECKS ---

    matches = []

    # Check 1: PFAS
    if mol.HasSubstructMatch(pfas_pattern):
        matches.append("PFAS-like (Fluorine chain)")

    # Check 2: Phthalates
    if mol.HasSubstructMatch(phthalate_pattern):
        matches.append("Phthalate-like structure")

    # Check 3: Heavy Halogenation (Persistent Organic Pollutants)
    # Count Cl (Chlorine) and Br (Bromine)
    num_cl = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Cl]")))
    num_br = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[Br]")))
    if num_cl + num_br >= 3:
        matches.append(f"High Halogen Content (Cl/Br count: {num_cl+num_br})")

    # Check 4: Bioaccumulation Potential (LogP)
    # Molecules with LogP > 5 are very fat-soluble and bioaccumulate in nature.
    log_p = Descriptors.MolLogP(mol)
    if log_p > 5.0:
         matches.append(f"High Bioaccumulation Risk (LogP: {log_p:.2f})")

    # --- RESULT ---
    
    if matches:
        logger.debug(f"Warning: Potential Environmental Pollutant Alerts: {matches}")
        return PollutionPotential.HIGH
    else:
        logger.debug("Low risk: No obvious environmental persistence alerts found.")
        return PollutionPotential.LOW

# --- TEST ---
@pytest.mark.parametrize("input,expected", [
    # Case 1: PMMA (Standard plastic, generally considered low risk monomer structure)
    ("*CC(*)(C)C(=O)OCC", PollutionPotential.LOW),
    
    # Case 2: PTFE (Teflon-like) - Should catch PFAS logic
    # Structure: ...-CF2-CF2-...
    ("FC(F)(*)C(F)(*)F", PollutionPotential.HIGH),
    
    # Case 3: A highly Chlorinated ring (like a PCB fragment)
    ("c1(Cl)c(Cl)c(Cl)ccc1", PollutionPotential.HIGH) 
])
def test_pollution(input, expected):
    result = check_pollution(input)
    assert result == expected, f"Input: {input} | Got: {result}"
