import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from typing import Optional
from loguru import logger

"""
NOTE! Made using AI.
NOTE! rdkit returns the units in Angstrom.

Takes a mol (from rdkit) -> Returns the mol with 3D coordinates.
"""
def transform_into_3D_geometry(mol: Chem.Mol) -> Optional[Chem.Mol]:
    mol = Chem.AddHs(mol) # CRITICAL: xTB needs explicit hydrogens
    
    # Generate 3D coordinates (Embedding) and do a quick classical optimization
    embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    
    # Fallback if standard embedding fails (returns -1)
    if embed_status == -1:
        logger.warning("ETKDGv3 embedding failed, attempting with random coordinates...")
        embed_status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        
        # If it STILL fails, we can't do QM on it.
        if embed_status == -1:
            logger.error("Failed to generate 3D conformer. Skipping.")
            return None
                
    AllChem.MMFFOptimizeMolecule(mol)
    return mol
    

import pytest
@pytest.mark.parametrize("input", [
    "CCCC",
    "CCCCCCCCCCCCCCCCCC(=O)O", # "Long" Chain
    "C1CC2CCC1C2", # Rigid Cage (Norbornane)
    "C[C@H](F)Cl", # Chiral Center
    "C[N+](C)(C)C", # Charged Species
])
def test_usage(input):
    print()
    mol = Chem.MolFromSmiles(input)
    mol_3D = transform_into_3D_geometry(mol)
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol_3D.GetAtoms()])
    coords_angstrom = mol_3D.GetConformer().GetPositions()
    print(f"atomic_numbers: {atomic_numbers}")
    print(f"coords_angstrom:\n{coords_angstrom}")
