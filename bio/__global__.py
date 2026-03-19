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
LOGURU_SIMPLE_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{message}</level>"

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
    "polyethylene": "*CC*",
    'Methacrylic acid': '*CC(C)(C(=O)O)*', # same as C-mA
    'Vinylpyridine': '*CC(c1ccncc1)*',
    'polyethyleneimine': '*CCN*',
    'Polyamide 4/6 Nanofibers': '*NCCCCNC(=O)CCCC(=O)*',
    
    'CP-APM': '*OC1C(O)C(O)C(*)OC1COC(=O)CC(O)(C(=O)O)CC(=O)NCCN', # MOST LIKELY THIS IS WRONG!
    'Poly(N-isopropylacrylamide) ferrogel': '*CC(C(=O)NC(C)C)*', # Just the polyemer, cannot represetn a ferrogel!
    
    "Graphene oxide (GO)": "",
    "Graphene oxide": "",
    "Magnetic graphene oxide functionalized with nitricacid(NDMGO)": "",
    "GO/sodium alginate hydrogels": "",
    "GO/calcium alginate fibers": "",
    "Porous graphene oxide-chitosan aerogel (PGO-CS)": "",
    "Fe3O4@SiO2 Chitosan/GO": "",
    "NDMGO": "",
    "Thiourea-dioxide-reduced magnetic graphene oxide (TDMGO)": "",
    "Diethylenetriaminepenta-acetic acid-functionalized magnetic graphene oxide (DDMGO)": "",
    "Pristine graphene": "",
    "Graphene oxide nanoparticles": "",
    "Graphene oxide-ionic liquids": "",
    "Nitrogen-doped reduced graphene oxide/Fe3O4 nanocomposite": "",
    "Magnetic genipin-crosslinked chitosan/graphene oxide-SO3H composite": "",
    "Magnetically modified graphene nanoplatelets": "",
    "Fe3O4 nanoparticle": "",
    "NiFe2O4/biochar composites": "",
    "Biochar/MgFe2O4": "",
    "Magnetic NPs coated with Rhamnolipids (Rh-cMNP)": "",
    "Magnetic mesoporous silica microspheres": "",
    "Iron, copper oxides composite particles (Fe/Cu oxides CPs)": "",
    "MgFe2O4": "",
    "MgO": "",
    "ZnO-MgO nanocomposites": "",
    "Fe3O4@C Matrix": "",
    "Fe3O4/Douglas fir biochar": "",
    "(α-Fe2O3)": "",
    "Magnetic Fe3O4 NPs coated zeolite": "",
    "MnO2@carbonmicrospheres": "",
    "Magnetic cadmiumbased MOFs modified with chitosan \n(Fe3O4@Cd-MOF@CS)": "",
    "Mn-Zn ferrite/biochar (MZF-BC)": "",
    "NiFe2O4-COF-chitosanterphthalaldehyde nanocomposites film": "",
    "Fe3O4/CD/AC/SA": "",
    "Fe3O4/red mud": "",
    "Amine-coated magnetic Nanocomposite NiFe2O4@SiO2": "",
    "Rape straw biomass fiber/β-Cyclodextrin/Fe3O4": "",
    "Plantain peel activated carbon-supported zinc oxide (PPAC-ZnO) nanocomposite": "",
    "Polyacrylonitrile/Polyaniline (PAN/PANI)": "*NC1=CC=C(C=C1)*", # This is the simplified repeating unit
    "Polypyrrole functionalized Calotropis gigantea fiber (PPy-O-CGF)": "",
    "PPy-PANI copolymer": "",
    "Poly(aniline-co-pyrrole)": "",
    "Functional Polyaniline/multiwalled carbon nanotube composite (PANI/MWCNT)": "",
    "Polyaniline-deposited cellulose fiber composite": "",
    "Poly(aniline-co-pyrrole)/multi-walled carbon nanotubes": "",
    "Pristine polypyrrole": "*c1ccc(*)[nH]1", 
    "Polypyrrole/Zinc oxide (PPy/ZnO)": "*c1ccc(*)[nH]1", # just PPy
    "Amberlite-XAD-16 polymer": "*CC(c1ccccc1)*", # This is the simplified repeating unit 
    "Amberlite-XAD-4 polymer": "*CC(c1ccccc1)*", # This is the simplified repeating unit
    "Poly(styrene-block-acrylic acid) deblock copolymer/Fe3O4 magnetic": "",
    "Polyaniline/Graphene Oxide Based Nanocomposites": "*NC1=CC=C(C=C1)*", # Just PANI
    "Polyaniline-coated magnetic nanoparticles": "*NC1=CC=C(C=C1)*", # Just PANI
    "Fe3O4 coated polymer (clay: chitosan) composite": "",
    "Chitosan-carbon nanotubes": "",
    "Poly(acrylic acid) grafted chitosan/Graphite oxide": "*CC(C(=O)O)*", # Just the PAA graft
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
    'Atenolol': '',
    'Propranolol': '',
    'Doxorubicin': '',
    'Ciprofloxacin': '',
    'Tetracycline': '',
    'Metformin': '',
    'Ketoprofen': '',
    'Norfloxacin': '',
    'Amoxicillin': '',
    'Carbamazepine': '',
    'Levofloxacin': '',
    'Acetaminophen': '',
    'Paracetamol': '',
    'Ofloxacin': '',
    'Chlorpyrifos': '',
    'Linezolid antibiotic': '',
    'Acetylsalicylic acid': '',
    'Cephalexin': '',
    'Cefotaxime': '',
    'Chloroquine for COVID-19 treatment': '',
    'Chloroxylenol': '',
    'N,N-diethyl-meta-toluamide': '',
    'Enrofloxacin': '',
    'Metronidazole': '',
    'Phenylbutazone': '',
    'Ceftiofur': '',
    'Prednisolone': '',
    'Meloxicam': '',
    'Anti-inflammatory': '', 
    'Gemfibrozil': '',
    'Dorzolamide': '',
}

assert BIOINFORMATICS_DIR.exists()
assert REPO_DIR.exists()
assert DATASETS_DIR.exists()
assert VOCABULARIES_DIR.exists()
assert PDCC_DATASET.exists()
# assert INTERPOLATED_CSV.exists()

def test_():
    pass

import pytest
@pytest.mark.skip(reason="I needed this once")
def test_specific_rows_data():
    """
    Test specific row numbers to ensure they contain expected data.
    Note: pandas is zero-indexed, so row 37 in a CSV might be index 35 or 36 depending on headers.
    Adjust the indices below if needed.
    """
    import pandas as pd
    my_dataframe = pd.read_csv(PDCC_CSV)
    target_rows = [37, 46, 55, 64, 73, 74, 75]
    
    # Check that the dataframe is long enough to contain these rows
    max_row = max(target_rows)
    assert len(my_dataframe) > max_row, f"DataFrame only has {len(my_dataframe)} rows, expected at least {max_row + 1}"
    
    # Extract the specific rows
    extracted_data = my_dataframe.iloc[target_rows]
    
    # Example assertion: Ensure none of these specific rows are completely empty
    # Replace this with whatever specific logic you usually test for!
    assert not extracted_data.empty, "The extracted rows are empty."
    
import pytest
@pytest.mark.skip(reason="I needed this once")
def test_no_nan_values_in_csv():
    """
    Test to ensure there are no NaN values anywhere in the dataset.
    If there are, fail the test and print their locations.
    """
    import pandas as pd
    my_dataframe = pd.read_csv(PDCC_CSV)
    # Check if there are any NaNs in the entire dataframe
    has_nans = my_dataframe.isna().any().any()
    
    if has_nans:
        # Find exactly where the NaNs are to make debugging easier
        nan_locations = my_dataframe[my_dataframe.isna().any(axis=1)]
        
        # Get a count of NaNs per column
        nan_counts = my_dataframe.isna().sum()
        nan_counts = nan_counts[nan_counts > 0]
        
        error_msg = (
            f"Found NaN values in the dataset!\n"
            f"Columns with NaNs and their counts:\n{nan_counts.to_string()}\n"
            f"Row indices containing NaNs: {nan_locations.index.tolist()}"
        )
        pytest.fail(error_msg)
