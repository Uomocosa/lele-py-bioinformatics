import numpy as np
from typing import List, Tuple
from rdkit.DataStructs import TanimotoSimilarity
import bio

"""
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
        "c1ccccc1"    # benzene
    ]

    diversity_list, mean_diversity = calculate_diversity(smiles)
    print(f"diversity_list: {diversity_list}")
    print(f"mean_diversity: {mean_diversity}")

    # There should be n*(n-1)/2 pairwise comparisons
    assert len(diversity_list) == 3

    # All diversity values should be between 0 and 1
    for d in diversity_list: assert 0.0 <= d <= 1.0

    # Mean diversity should also be between 0 and 1
    assert 0.0 <= mean_diversity <= 1.0

    # Mean should equal numpy mean of the list
    assert np.isclose(mean_diversity, np.mean(diversity_list))
