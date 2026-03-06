import pubchempy as pcp
from bio.__global__ import SMILES_DICT, PSMILES_DICT
from loguru import logger


def test_():
    # d = {s: get_smiles_from_name(s) for s in SMILES_DICT.keys()}
    # print(d)

    # get_smiles_from_name('2-benzhydryl-4,5-dihydro-1,3-oxazole') # C1COC(=N1)C(C2=CC=CC=C2)C3=CC=CC=C3
    # get_smiles_from_name('Granular Activated Carbon') # Not a polymer
    # get_smiles_from_name('Bagasse Fly Ash') # Not a polymer
    # get_smiles_from_name('Quinoline') # C1=CC=C2C(=C1)C=CC=N2
    # get_smiles_from_name('SMIP/MCNSs') # polymeric nanocomposites
    # get_smiles_from_name('SNIP/MCNSs') # polymeric nanocomposites
    # get_smiles_from_name('Sodium aluminosilicate') # O.[O-2].[O-2].[O-2].[O-2].[O-2].[O-2].[Na+].[Na+].[Al+3].[Al+3].[Si+4]
    # get_smiles_from_name('Cadmium chloride') # Cl[Cd]Cl
    # get_smiles_from_name('Geopolymer GP2M') # No exact match found
    # get_smiles_from_name('GP2M') # No exact match found
    # get_smiles_from_name('fluoxetine') # CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F
    # get_smiles_from_name('propranolol') # CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O
    # get_smiles_from_name('ketamine') # CNC1(CCCCC1=O)C2=CC=CC=C2Cl
    # get_smiles_from_name('atorvastatin') # CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4
    # get_smiles_from_name('carbamazepine') # C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N
    # get_smiles_from_name('ethylene') #  C=C
    # get_smiles_from_name('diclofenac') #  C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl
    # get_smiles_from_name('naproxen') #  C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O
    # get_smiles_from_name('Methacrylic acid') #  CC(=C)C(=O)O
    # get_smiles_from_name('Vinylpyridine') #  C=CC1=CC=CC=N1
    # get_smiles_from_name('N-isopropylacrylamide') #  CC(C)NC(=O)C=C
    # get_smiles_from_name('Cd(II)') # [Cd+2]
    # get_smiles_from_name('Pb(II)') # [Pb+2]
    # get_smiles_from_name('Silver') # [Ag]
    # get_smiles_from_name('Congo red') # C1=CC=C2C(=C1)C(=CC(=C2N)N=NC3=CC=C(C=C3)C4=CC=C(C=C4)N=NC5=C(C6=CC=CC=C6C(=C5)S(=O)(=O)[O-])N)S(=O)(=O)[O-].[Na+].[Na+]
    # get_smiles_from_name('ethyleneimine') # C1CN1
    get_smiles_from_name('Clarithromycin') # CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)O)(C)O
    get_smiles_from_name('Amoxicillin trihydrate') # 
    get_smiles_from_name('Sulfamethoxazole') # 
    get_smiles_from_name('Trimethoprim') # 
    get_smiles_from_name('Azithromycin dihydrate') # 
    # get_smiles_from_name('') # 
    


def get_smiles_from_name(name: str) -> str:
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            # results[0] gets the top match. We request the absolute SMILES.
            logger.debug(f"{name}: {results[0].smiles}")
            return results[0].smiles
        else:
            logger.error(f"{name}: No exact match found")
            return ""
    except Exception as e:
        logger.error(f"Error fetching {name}: {e}")
