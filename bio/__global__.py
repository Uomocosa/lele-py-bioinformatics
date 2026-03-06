import lele

BIOINFORMATICS_DIR = lele.P(".")
REPO_DIR = BIOINFORMATICS_DIR.parent
DATASETS_DIR = BIOINFORMATICS_DIR/'DATASETS'
VOCABULARIES_DIR = BIOINFORMATICS_DIR/'VOCABULARIES'
PDCC_DATASET = DATASETS_DIR / "PDCC" / "polymer_drug_concentration_capacity.csv"
INTERPOLATED_CSV = DATASETS_DIR / "PDCC" / "interpolated_pdcc.csv"
PDCC_CSV = PDCC_DATASET
INTERPOLATED_PDCC_CSV = INTERPOLATED_CSV
CONVERTED_PDCC_CSV = DATASETS_DIR / "PDCC" / "converted_pdcc.csv"

PSMILES_DICT = {
    # First I used https://decimer.ai/ to get the SMILES representation from the papers images.
    # Then I asked AI to transform it into a P-SMILES string. This passage especially can bring many errors.
    "C-megl": "*CC(C1=CC=C(C=C1)CN(C)C[C@@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O)*",
    "C-lys": "*CC(C)(C(=O)OCCOC(=O)C(CCN)N)*",
    "C-mA": "*CC(C)(C(=O)O)*", # C=C(C)C(=O)O
    "C-ph": "*CC(C)(C(=O)OCCO)*", # C=C(C)C(=O)OCCO
    
    # THIS WAS TAKEN OUT OF THE AIR!!! 
    # TODO!!! HOW DO WE FIND THIS???
    # "polyPhOx": "*N(C(=O)C(c1ccccc1)c2ccccc2)CC*",
    # "polyPhOx": "*N(C(=O)c1ccccc1)CC*",
    # "polyPhOx": "*C1COC(=N1)C(C2=CC=CC=C2)C3=CC=CC=C3*",
    "polyPhOx": "*CN(CCO)C(=O)C(c1ccccc1)c2ccccc2*", # dedotto dal paper CACCA
    "polyethylene": "*C=C*",
    'Methacrylic acid': '*CC(=C)C(=O)O*',
    'Vinylpyridine': '*C=CC1=CC=CC=N1*',
    'polyethyleneimine': '*C1CN1*',
    'Polyamide 4/6 Nanofibers': '*NCCCCNC(=O)CCCC(=O)*',
    
    # UNDER THIS LINE THERE ARE ONLY "WRONG P-SMILES!"
    'CP-APM': '*OC1C(O)C(O)C(*)OC1COC(=O)CC(O)(C(=O)O)CC(=O)NCCN', # MOST LIKELY THIS IS WRONG!
    'Poly(N-isopropylacrylamide) ferrogel': '*CC(C(=O)NC(C)C)*', # THIS IS WRONG (not a ferrogel)!
}


SMILES_DICT = { 
    # We are sure about these. 
    # Gotten from get_smiles_from_name.
    # Can also be found on Wikipedia
    'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O', 
    'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 
    'Indomethacin': 'CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O', 
    'Metoclopramide': 'CCN(CC)CCNC(=O)C1=CC(=C(C=C1OC)N)Cl', 
    'Oestradiol': 'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O', 
    'Pyramidone': 'CC1=C(C(=O)N(N1C)C2=CC=CC=C2)N(C)C', 
    '2.4D': 'Clc1cc(Cl)ccc1OCC(=O)O', # 2,4-Dichlorophenoxyacetic acid
    'Ampicillin': 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C', 
    'Doxycycline': 'C[C@@H]1[C@H]2[C@@H]([C@H]3[C@@H](C(=O)C(=C([C@]3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)O)C(=O)N)N(C)C)O', 
    'Fluconazole': 'C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O', 
    'Lomefloxacin': 'CCN1C=C(C(=O)C2=CC(=C(C(=C21)F)N3CCNC(C3)C)F)C(=O)O', 
    'Methylene Violet': 'CN(C)C1=CC2=C(C=C1)N=C3C=CC(=O)C=C3S2', 
    'Moxidectin': 'C[C@@H]\\1C/C(=C/C[C@@H]2C[C@@H](C[C@@]3(O2)C/C(=N\\OC)/[C@@H]([C@H](O3)/C(=C/C(C)C)/C)C)OC(=O)[C@@H]4C=C([C@H]([C@@H]5[C@]4(/C(=C/C=C1)/CO5)O)O)C)/C', 
    'Piroxicam': 'CN1C(=C(C2=CC=CC=C2S1(=O)=O)O)C(=O)NC3=CC=CC=N3', 
    'Thymol Blue': 'CC1=CC(=C(C=C1C2(C3=CC=CC=C3S(=O)(=O)O2)C4=CC(=C(C=C4C)O)C(C)C)C(C)C)O',
    # 'Quinoline': 'C1=CC=C2C(=C1)C=CC=N2',
    'fluoxetine': 'CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F',
    'propranolol': 'CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O',
    'ketamine': 'CNC1(CCCCC1=O)C2=CC=CC=C2Cl',
    'atorvastatin': 'CC(C)C1=C(C(=C(N1CC[C@H](C[C@H](CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4',
    'carbamazepine': 'C1=CC=C2C(=C1)C=CC3=CC=CC=C3N2C(=O)N',
    'diclofenac': 'C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl',
    'naproxen': 'C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O',
    'Cd(II)': '[Cd+2]',
    'Pb(II)': '[Pb+2]',
    'Silver': '[Ag]',
    'Congo red': 'C1=CC=C2C(=C1)C(=CC(=C2N)N=NC3=CC=C(C=C3)C4=CC=C(C=C4)N=NC5=C(C6=CC=CC=C6C(=C5)S(=O)(=O)[O-])N)S(=O)(=O)[O-].[Na+].[Na+]',
    'Clarithromycin': 'CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)O)(C)O',
    'Amoxicillin trihydrate': 'CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=C(C=C3)O)N)C(=O)O)C.O.O.O',
    'Sulfamethoxazole': 'CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N',
    'Trimethoprim': 'COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N',
    'Azithromycin dihydrate': 'CC[C@@H]1[C@@]([C@@H]([C@H](N(C[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O.O.O',
}

assert BIOINFORMATICS_DIR.exists()
assert REPO_DIR.exists()
assert DATASETS_DIR.exists()
assert VOCABULARIES_DIR.exists()
assert PDCC_DATASET.exists()
# assert INTERPOLATED_CSV.exists()

def test_():
    pass
