import numpy as np
from typing import List, Tuple
from rdkit.DataStructs import TanimotoSimilarity
import bio
from loguru import logger

"""
Computes pairwise Tanimoto similarity between all unique fingerprint pairs
Converts similarity to diversity using 1 − similarity
Returns both the mean diversity of a molecule from the others.

From the paper:
Diversity: provides a dissimilarity score among the generated polymers, evaluating whether the generated
polymers are diverse so that no mode collapse issue is witnessed in the generation. For all the metrics, a higher
value indicates better performance. These metrics or similar metrics have been utilized in various prior studies
for evaluating molecule generation tasks[47]. A detailed quantitative breakdown of how these metrics are
computed can be found in the Methods section.
"""
def calculate_mean_diversity(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    logger.warning("NEEDS REVISION!")
    fingerprints = bio.PolyGen.calculate_morgan_fingerprint(smiles_list)
    n = len(smiles_list)
    diversity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            diversity_matrix[i, j] = 1 - similarity
            diversity_matrix[j, i] = 1 - similarity
            
    # diversity_mean_list: Average distance of each molecule to all others
    # We divide by (n-1) to exclude the molecule's comparison to itself (0.0)
    if n > 1: diversity_mean_list = np.sum(diversity_matrix, axis=1) / (n - 1)
    else: diversity_mean_list = np.array([0.0])
    return diversity_mean_list


def test_1():
    smiles = [
        "CCO",        # ethanol
        "CCCO",       # propanol
        "c1ccccc1",    # benzene
        "c1ccccc1",    # benzene
        "c1ccccc1",    # benzene
    ]
    diversity = calculate_mean_diversity(smiles)
    print(f"diversity: {diversity}")



def test_2():
    print(); print();
    smiles = [
        "CCO",        # ethanol
        "CCCO",       # propanol
        "c1ccccc1",    # benzene
        "C[C@@]1(C(=O)C=C(O1)C(=O)[O-])c2ccccc2",    # valid SMILE from ZINC_base
        "c1ccc(cc1)C(c2ccccc2)[S@](=O)CC(=O)NO",    # valid SMILE from ZINC_base
        "CCC(=O)O[C@]1(CC[NH+](C[C@@H]1CC=C)C)c2ccccc2",    # valid SMILE from ZINC_base
        "C[C@@H](CC(c1ccccc1)(c2ccccc2)C(=O)N)[NH+](C)C",    # valid SMILE from ZINC_base
        "Cc1c(c(=O)n(n1C)c2ccccc2)NC(=O)[C@H](C)[NH+](C)C",    # valid SMILE from ZINC_base
        "c1ccc(cc1)[C@@H](C(=O)[O-])O",    # valid SMILE from ZINC_base
        "CC[C@](C)(C[NH+](C)C)OC(=O)c1ccccc1",    # valid SMILE from ZINC_base
    ]
    diversity = calculate_mean_diversity(smiles)
    print(f"diversity: {diversity}")
