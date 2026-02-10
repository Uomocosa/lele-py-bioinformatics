import sys
import io
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

"""
NOTE! Refactored using GEMINI (AI)
NOTE! This function is useless for my application

This function acts as a structural validator specifically designed 
for molecules meant to have a linear or end-capped geometry 
involving copper and gold.

In short, it ensures that each molecule is a valid chemical entity and 
follows a "one-copper-one-gold" terminal rule—meaning the 
metals must appear exactly once and only at the ends of the molecular 
chain.
"""
def get_validity_label(smiles, column_name):
    """
    NOTE! This function is useless for my application
    """
    if pd.isna(smiles): return "none"
    
    RDLogger.DisableLog('rdApp.*') # disable warnings
    mol = Chem.MolFromSmiles(smiles)

    if mol is None: return "invalid smile"

    # Structural Checks (Cu/Au specific)
    # 1. Count check (Must have exactly one Cu and one Au)
    if (smiles.count("[Cu]") != 1) or (smiles.count("[Au]") != 1):
        return 'More than two ends'

    # 2. Bond type checks (Looking for double/triple bonds at metal sites)
    # Using string search for efficiency as per original logic, 
    # though RDKit Atom/Bond objects are more robust.
    if any(x in smiles for x in ['=[Cu]', '[Cu]=', '=[Au]', '[Au]=']):
        return 'Double bond at the end point'
    
    if any(x in smiles for x in ['#[Cu]', '[Cu]#', '#[Au]', '[Au]#']):
        return 'Triple bond at the end point'

    # 3. Degree check (Metal atoms should only have one bond)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ["Cu", "Au"]:
            if atom.GetDegree() > 1:
                return 'More than one bonds at the end point'

    return 'ok'

def validate_mol(df, column_name):
    """
    Validates molecular SMILES in a dataframe and adds a 'validity' column.
    """
    df = df.copy() # Avoid SettingWithCopyWarning
    df['validity'] = df[column_name].apply(lambda x: get_validity_label(x, column_name))
    return df


def test_validate_mol():
    test_data = {
        'mol_smiles': [
            'CCO[Cu].CC[Au]',          # ok
            'invalid_smiles',          # RDKit error
            None,                      # none
            'C=C=[Cu].C[Au]',          # Double bond at the end point
            'C#[Cu].C[Au]',            # Triple bond at the end point
            'C[Cu].C[Cu].C[Au]',       # More than two ends
            'C1(O[Cu])CC1(O[Au])C'     # Degree > 1 (if branched at metal - logic check)
        ]
    }
    df = pd.DataFrame(test_data)
    
    # Run validation
    df_results = validate_mol(df, 'mol_smiles')
    
    print("\nValidation Results:")
    print(df_results)

    # Simple assertions
    assert df_results.iloc[0]['validity'] == 'ok'
    assert df_results.iloc[2]['validity'] == 'none'
    assert 'More than two ends' in df_results.values


"""
The follwoing is the original function
"""
# import sys
# import io
# import pandas as pd
# from rdkit import Chem
# from rdkit import rdBase

# def validate_mol(mol_list, column_name):
#     sio = sys.stderr = StringIO()
#     for i in mol_list['mol_smiles']:

#         if pd.isna(i):
#             mol_list.loc[mol_list[column_name] == i, 'validity']  = "none"
#         elif Chem.MolFromSmiles(i) is None:
#             mol_list.loc[mol_list[column_name] == i, 'validity']  = sio.getvalue().strip()[11:]
#             sio = sys.stderr = StringIO() # reset the error logger
#         elif ('=[Cu]' in i) or ('[Cu]=' in i) or ('=[Au]' in i) or ('[Au]=' in i):
#             mol_list.loc[mol_list[column_name] == i, 'validity']  = 'Double bond at the end point'
#         elif ('#[Cu]' in i) or ('[Cu]#' in i) or ('#[Au]' in i) or ('[Au]#' in i):
#             mol_list.loc[mol_list[column_name] == i, 'validity']  = 'Triple bond at the end point'
#         elif (i.count("[Cu]") != 1) or (i.count("[Au]") != 1):
#             mol_list.loc[mol_list[column_name] == i, 'validity']  = 'More than two ends'
#         else:
#             bond_flag = False
#             for atom in Chem.MolFromSmiles(i).GetAtoms():
#                 if atom.GetSymbol() == "Cu":
#                     if atom.GetDegree() > 1:
#                         bond_flag = True
#                 elif atom.GetSymbol() == "Au":
#                     if atom.GetDegree() > 1:
#                         bond_flag = True
#             if bond_flag:
#                 mol_list.loc[mol_list[column_name] == i, 'validity']  = 'More than one bonds at the end point'
#             else:
#                 mol_list.loc[mol_list[column_name] == i, 'validity'] = 'ok'
#     return mol_list



# def test_():
#     pass
