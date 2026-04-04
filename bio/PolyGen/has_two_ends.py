import pandas as pd
import numpy as np


"""
NOTE! Refactored using GEMINI (AI)

Identifies "hetero-telechelic" structures by verifying the presence of exactly 
two specific terminal caps. It returns True only if the SMILES string contains 
exactly one Copper ([Cu]) and exactly one Gold ([Au]).
"""
def has_two_ends(df, column_name='mol_smiles'):
    """
    Checks if each SMILES has exactly one [Cu] and one [Au].
    """
    # Create a copy to avoid modifying the original dataframe in place
    df = df.copy()
    
    # Handle NaN values by filling with empty strings for the count check
    smiles_series = df[column_name].fillna('')
    
    # Vectorized counting: Much faster than a for-loop
    count_cu = smiles_series.str.count(r'\[Cu\]') == 1
    count_au = smiles_series.str.count(r'\[Au\]') == 1
    
    # Combine conditions: True only if both are exactly 1
    df['has_two_ends'] = count_cu & count_au
    
    # Set NaN rows back to False or NaN if preferred (here we'll keep them False)
    df.loc[df[column_name].isna(), 'has_two_ends'] = False
    
    return df


def test_():
    test_data = {
        'mol_smiles': [
            'C[Cu].C[Au]',      # True: One of each
            'C[Cu].C[Cu]',      # False: Two Cu, zero Au
            'C[Au]',            # False: Zero Cu, one Au
            'C[Cu].C[Au].C[Au]', # False: One Cu, two Au
            None,                # False: NaN
            'CCCC'               # False: No metals
        ]
    }
    df = pd.DataFrame(test_data)
    df_results = has_two_ends(df)
    
    print("Test Results:")
    print(df_results)
    
    # Assertions
    assert df_results.iloc[0]['has_two_ends'] == True
    assert df_results.iloc[1]['has_two_ends'] == False
    assert df_results.iloc[4]['has_two_ends'] == False
