from typing import List
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ExplicitBitVect

"""
Similarity calculation based on TanimotoSimilarity
"""
def calculate_morgan_fingerprint(
    smiles_list: List[str],
    radius = 2,  # Morgan fingerprint radius
    n_bits = 2048,  # Number of bits in the fingerprint
) -> List[ExplicitBitVect]:
    fp_lst = []
    morgan_generator = GetMorganGenerator(
        radius=radius,
        fpSize=n_bits
    )
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        fingerprint = morgan_generator.GetFingerprint(mol)
        fp_lst.append(fingerprint)
    return fp_lst


def test_():
    fps = calculate_morgan_fingerprint(["CCO", "CC"])
    print(f"fps: {fps}")
    print(f"fps[0]: {fps[0]}")
    # print(f"dir(fps[0]): {dir(fps[0])}")
    # print(f"fps[0].ToBitString(): {fps[0].ToBitString()}")
    print(f"fps[0].ToBase64(): {fps[0].ToBase64()}")
    assert isinstance(fps[0], ExplicitBitVect)
