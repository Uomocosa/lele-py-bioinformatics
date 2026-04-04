import math
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from typing import Optional
import bio
from bio.PolyGen.__global__ import FPSCORES

"""
NOTE! Refactored using GEMINI (AI)

FUNCTION SUMMARY:
Calculates the Synthetic Accessibility (SA) Score for a molecule.
Range: 1 (Very Easy) to 10 (Very Difficult).
Logic:
- Fragment Score: Uses Morgan Fingerprints to check fragment popularity.
- Complexity Penalties: Penalizes large size, stereo centers, and complex ring junctions.
- Normalization: Maps raw values to a human-readable 1-10 scale.
"""

def calculate_sa_score(mol: Chem.Mol) -> Optional[float]:
    if not mol: return None
    RDLogger.DisableLog('rdApp.*')
    
    # Ensure fragment scores are loaded into the global FPSCORES
    global FPSCORES
    if FPSCORES is None: FPSCORES = bio.PolyGen.read_fragment_scores()

    # 1. Fragment Score calculation
    # Morgan Fingerprint radius 2 is equivalent to ECFP4
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bit_id, count in fps.items():
        nf += count
        # Default to -4 for "unseen/rare" fragments
        score1 += FPSCORES.get(bit_id, -4) * count
    score1 /= nf

    # 2. Complexity Penalties (Features Score)
    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    n_bridgeheads, n_spiro = bio.PolyGen.num_bridgeheads_and_spiro(mol) # Using our refactored helper
    
    # Count macrocycles (rings larger than 8 atoms)
    n_macrocycles = sum(1 for x in ri.AtomRings() if len(x) > 8)

    # Calculate individual penalties
    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    
    macrocycle_penalty = 0.
    if n_macrocycles > 0:
        macrocycle_penalty = math.log10(2) # Non-linear penalty for macrocycles

    score2 = 0. - size_penalty - stereo_penalty - spiro_penalty - bridge_penalty - macrocycle_penalty

    # 3. Symmetry Correction (Fingerprint Density)
    score3 = 0.
    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    # Total Raw Score
    sascore = score1 + score2 + score3

    # 4. Normalization to 1-10 Scale
    min_score, max_score = -4.0, 2.5
    sascore = 11. - (sascore - min_score + 1) / (max_score - min_score) * 9.

    # Smoothing the upper bound
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    
    return max(1.0, min(10.0, sascore))


import pytest
@pytest.mark.parametrize("smile, expected", [
    ( # Ethanol is easy
        "CCO", 
        1.980
    ), 
    
    ( # Taxol is hard
        "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)O",
        5.823
    ),
])
def test_sa_score(smile, expected):
    mol = Chem.MolFromSmiles(smile)
    assert mol is not None, f"{valid_smile} is an INVALID SMILE"
    score = calculate_sa_score(mol)
    assert pytest.approx(score, rel=1e-3) == expected, f"Expected {expected}, but got {score}"
