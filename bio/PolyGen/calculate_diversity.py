import numpy as np
from typing import List, Tuple
from rdkit.DataStructs import TanimotoSimilarity
from loguru import logger
import bio

"""
NOTE! This retunrns a semimatrix, and a mean, not sure what they wanted to get here!
NOTE! Deprecated use bio.Metrics.calculate_mean_diversity

Computes pairwise Tanimoto similarity between all unique fingerprint pairs
Converts similarity to diversity using 1 − similarity
Returns both the list of pairwise diversity values and their mean
"""
def calculate_diversity(smiles_list: List[str]) -> Tuple[List[float], float]:
    fingerprints = bio.PolyGen.calculate_morgan_fingerprint(smiles_list)
    diversity_list = []
    for i in range(len(smiles_list)):
        for j in range(i):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            diversity_list.append(1-similarity)
    return diversity_list, np.mean(diversity_list)


def test_():
    smiles = [
        "CCO",        # ethanol
        "CCCO",       # propanol
        "c1ccccc1",    # benzene
        "c1ccccc1",    # benzene
        "c1ccccc1",    # benzene
        "CCCO",       # propanol
    ]
    diversity_list, mean_diversity = calculate_diversity(smiles)
    print(f"diversity_list: {diversity_list}")
    print(f"mean_diversity: {mean_diversity}")
