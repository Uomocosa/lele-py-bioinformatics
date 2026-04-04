import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from loguru import logger

def transform_into_fingerprint(mol: Chem.Mol, radius=2, nBits=1024):
    logger.warning('DEPRECATED: use AllChem.GetMorganFingerprintAsBitVect directly')
    if not mol: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return fp
