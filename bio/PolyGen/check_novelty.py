
import numpy as np
import pandas as pd

"""
NOTE! Refactored using GEMINI (AI)

Compares generated SMILES against the training set.
Labels them as 'novel' if they are not found in the training data.
"""
def check_novelty(df_generated, df_train, column_name):
    # Suggested by GEMINI (AI), it uses the .isin method
    is_not_novel = df_generated[column_name].isin(df_train[column_name])
    df_generated['diversity'] = np.where(is_not_novel, 'In the original data set', 'novel')
    return df_generated



def test_():
    df_train = pd.DataFrame({'smiles': ['CCO', 'CCOC', 'CCOCC']})
    df_generated = pd.DataFrame({'smiles': ['CCO', 'C1=CC=CC=C1']})
    df_generated = check_novelty(df_generated, df_train, 'smiles')
    print(df_generated)
    assert df_generated['diversity'].iloc[0] == 'In the original data set'
    assert df_generated['diversity'].iloc[1] == 'novel'


# """
# This is the original function used in the PolyGen repo
# """
# def original_function(df_generated, df_train, column_name):
#     for i in df_generated[column_name]:
#         if df_train[column_name].eq(i).any():
#             df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'In the original data set'
#         else:
#             df_generated.loc[df_generated[column_name] == i, 'diversity'] = 'novel'
#     return df_generated
