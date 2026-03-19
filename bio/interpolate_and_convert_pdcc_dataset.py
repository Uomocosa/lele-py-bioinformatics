import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import bio
from loguru import logger
from bio.__global__ import PDCC_CSV, INTERPOLATED_PDCC_CSV, CONVERTED_PDCC_CSV, PSMILES_DICT, SMILES_DICT


@dataclass
class SplittedDataFrame:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def main():
    n_points = 2
    assert PDCC_CSV.exists()
    df = pd.read_csv(PDCC_CSV)
    df = interpolate(
        df, 
        n_points, 
        # interpolate_fn=PchipInterpolator
    )
    # NOTE! If we add origin points, before interpolatation, 
    #       we'll add many more data points, not sure if we want this!
    df = add_origin_points(df)
    logger.debug(df)
    PDCC_DIR = INTERPOLATED_PDCC_CSV.parent
    # df.to_csv(INTERPOLATED_PDCC_CSV, index=False)
    converted_df = df.copy()
    converted_df = convert(converted_df, PSMILES_DICT, SMILES_DICT)
    converted_df.to_csv(CONVERTED_PDCC_CSV, index=False)

    splitted_df = polymer_division(
        df, 
        train_polymers = [
            'polyPhOx', 
            'polyethylene', 
            'Methacrylic acid', 
            'Vinylpyridine',
            'polyethyleneimine',
            'Polyamide 4/6 Nanofibers',
            'CP-APM',
            'Poly(N-isopropylacrylamide) ferrogel',
        ],
        validation_polymers = [
            'C-lys', 
            'C-ph', 
            'C-mA',
        ],
        test_polymers = [
            'C-megl',
        ],
    )
    splitted_df.train = splitted_df.train[splitted_df.train['SOURCE'] != 'interpolated'].copy()
    splitted_df.test = splitted_df.test.drop(columns=['SOURCE'], errors='ignore')
    splitted_df.validation = splitted_df.validation.drop(columns=['SOURCE'], errors='ignore')
    splitted_df.train = splitted_df.train.drop(columns=['SOURCE'], errors='ignore')
    splitted_df.train = convert(splitted_df.train, PSMILES_DICT, SMILES_DICT)
    splitted_df.validation = convert(splitted_df.validation, PSMILES_DICT, SMILES_DICT)
    splitted_df.test = convert(splitted_df.test, PSMILES_DICT, SMILES_DICT)
    logger.debug(f"splitted_df.train:\n{splitted_df.train}")
    splitted_df.train.to_csv(PDCC_DIR / "converted_pdcc_train.csv", index=False)
    splitted_df.validation.to_csv(PDCC_DIR / "converted_pdcc_validation.csv", index=False)
    splitted_df.test.to_csv(PDCC_DIR / "converted_pdcc_test.csv", index=False)


def polymer_division(
    df: pd.DataFrame, 
    train_polymers: list[str],
    validation_polymers: list[str],
    test_polymers: list[str],
) -> SplittedDataFrame:
    unique_polymers = set(df['POLYMER_USED'].unique())
    logger.debug(f"unique_polymers found: {len(unique_polymers)}\nunique_polymers: {list(unique_polymers)}")
    
    train_set = set(train_polymers)
    val_set = set(validation_polymers)
    test_set = set(test_polymers)
    assert not train_set.intersection(val_set), "Overlap found between train and validation polymers!"
    assert not train_set.intersection(test_set), "Overlap found between train and test polymers!"
    assert not val_set.intersection(test_set), "Overlap found between validation and test polymers!"
    
    train_df = df[df['POLYMER_USED'].isin(train_polymers)].copy()
    val_df = df[df['POLYMER_USED'].isin(validation_polymers)].copy()
    test_df = df[df['POLYMER_USED'].isin(test_polymers)].copy()
    total_rows = len(df)
    debug_msg  = f"\nActual row sizes - Train: {len(train_df)} ({len(train_df)/total_rows:.1%})"
    debug_msg += f"\nVal: {len(val_df)} ({len(val_df)/total_rows:.1%})"
    debug_msg += f"\nTest: {len(test_df)} ({len(test_df)/total_rows:.1%})"
    logger.debug(debug_msg)
    logger.debug(f"Number of unique groups in Train: {len(train_polymers)} -> {len(train_df)} rows")
    logger.debug(f"Number of unique groups in Val: {len(validation_polymers)} -> {len(val_df)} rows")
    logger.debug(f"Number of unique groups in Test: {len(test_polymers)} -> {len(test_df)} rows")
    return SplittedDataFrame(
        train = train_df, 
        validation = val_df, 
        test = test_df
    )



def group_automatic_division(
    df: pd.DataFrame, 
    group_labels: list[str],
    test_validation_train_percentages = [0.70, 0.15, 0.15],
    seed: int = 42
) -> SplittedDataFrame:
    assert math.isclose(sum(test_validation_train_percentages), 1), f"sum(test_validation_train_percentages = {test_validation_train_percentages}) must be equal to 1"
    trn_pct, val_pct, tst_pct = test_validation_train_percentages
    df['GROUP'] = df[group_labels].astype(str).agg('_'.join, axis=1)
    unique_groups = df['GROUP'].unique()
    logger.debug(f"GROUPS found: {len(unique_groups)}\nunique_groups: {unique_groups}")
    group_counts = df['GROUP'].value_counts().to_dict()
    sorted_groups = sorted(group_counts.keys(), key=lambda g: group_counts[g], reverse=True)
    total_rows = len(df)
    targets = {
        'train': total_rows * trn_pct,
        'validation': total_rows * val_pct,
        'test': total_rows * tst_pct
    }
    
    # Greedy allocation: give the next group to whichever split needs it most
    current_sizes = {'train': 0, 'validation': 0, 'test': 0}
    allocations = {'train': [], 'validation': [], 'test': []}
    for group in sorted_groups:
        size = group_counts[group]
        deficits = {k: targets[k] - current_sizes[k] for k in targets.keys()}
        best_bin = max(deficits, key=deficits.get)
        allocations[best_bin].append(group)
        current_sizes[best_bin] += size
    train_df = df[df['GROUP'].isin(allocations['train'])].copy()
    val_df = df[df['GROUP'].isin(allocations['validation'])].copy()
    test_df = df[df['GROUP'].isin(allocations['test'])].copy()
    train_df = train_df.drop(columns=['GROUP'])
    val_df = val_df.drop(columns=['GROUP'])
    test_df = test_df.drop(columns=['GROUP'])
    logger.debug(f"Target row sizes - Train: {targets['train']:.1f}, Val: {targets['validation']:.1f}, Test: {targets['test']:.1f}")
    logger.debug(f"Actual row sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.debug(f"Number of unique groups in Train: {len(allocations['train'])} -> {len(train_df)}")
    logger.debug(f"Number of unique groups in Val: {len(allocations['validation'])} -> {len(val_df)}")
    logger.debug(f"Number of unique groups in Test: {len(allocations['test'])} -> {len(test_df)}")
    return SplittedDataFrame(
        train = train_df, 
        validation = val_df, 
        test = test_df
    )
    
    
def group_shuffle(df: pd.DataFrame, group_labels: list[str]) -> SplittedDataFrame:
    df['GROUP'] = df[group_labels].astype(str).agg('_'.join, axis=1)
    gss_train_valtest = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, valtest_idx = next(gss_train_valtest.split(df, groups=df['GROUP']))
    train_df = df.iloc[train_idx]
    valtest_df = df.iloc[valtest_idx]
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(valtest_df, groups=valtest_df['GROUP']))
    val_df = valtest_df.iloc[val_idx]
    test_df = valtest_df.iloc[test_idx]
    train_df = train_df.drop(columns=['GROUP'])
    val_df = val_df.drop(columns=['GROUP'])
    test_df = test_df.drop(columns=['GROUP'])
    logger.debug(f"Total dataset size: {len(df)}")
    logger.debug(f"Training set size: {len(train_df)}")
    logger.debug(f"Validation set size: {len(val_df)}")
    logger.debug(f"Test set size: {len(test_df)}")
    return SplittedDataFrame(
        train = train_df, 
        validation = val_df, 
        test = test_df
    )
    
def group_shuffle(df: pd.DataFrame, group_lables: list[str]) -> SplittedDataFrame:
    df['GROUP'] = df[group_labels].astype(str).agg('_'.join, axis=1)
    gss_train_valtest = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, valtest_idx = next(gss_train_valtest.split(df, groups=df['GROUP']))
    train_df = df.iloc[train_idx]
    valtest_df = df.iloc[valtest_idx]
    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(valtest_df, groups=valtest_df['GROUP']))
    val_df = valtest_df.iloc[val_idx]
    test_df = valtest_df.iloc[test_idx]
    train_df = train_df.drop(columns=['GROUP'])
    val_df = val_df.drop(columns=['GROUP'])
    test_df = test_df.drop(columns=['GROUP'])
    logger.debug(f"Total dataset size: {len(df)}")
    logger.debug(f"Training set size: {len(train_df)}")
    logger.debug(f"Validation set size: {len(val_df)}")
    logger.debug(f"Test set size: {len(test_df)}")
    return SplittedDataFrame(
        train = train_df, 
        validation = val_df, 
        test = test_df
    )
    
def convert(df: pd.DataFrame, psmiles_dict: dict, smiles_dict: dict):
    psmiles_dict_lower = {str(k).lower(): v for k, v in psmiles_dict.items()}
    smiles_dict_lower = {str(k).lower(): v for k, v in smiles_dict.items()}
    
    poly_lower = df['POLYMER_USED'].astype(str).str.lower()
    drug_lower = df['DRUG'].astype(str).str.lower()
    
    missing_polymers = {p for p in poly_lower if is_missing_or_empty(p, psmiles_dict_lower)}
    missing_drugs = {d for d in drug_lower if is_missing_or_empty(d, smiles_dict_lower)}
    
    if missing_polymers: logger.warning(f"Polymers missing from PSMILES_DICT!:\n{missing_polymers}")
    if missing_drugs: logger.warning(f"Molecules missing from SMILES_DICT!:\n{missing_drugs}")

    df['POLYMER_USED'] = poly_lower.map(psmiles_dict_lower)
    df['DRUG'] = drug_lower.map(smiles_dict_lower)
    
    # Convert empty strings (""), spaces ("   "), and literal "nan" strings to true np.nan
    df['POLYMER_USED'] = df['POLYMER_USED'].replace(r'^\s*$', np.nan, regex=True).replace(['nan', 'NaN', 'None'], np.nan)
    df['DRUG'] = df['DRUG'].replace(r'^\s*$', np.nan, regex=True).replace(['nan', 'NaN', 'None'], np.nan)
    df = df.dropna()
    return df


def is_missing_or_empty(item, mapping_dict):
    # 1. Ignore items that are already NaN, empty, or literal "nan" in the dataset
    if pd.isna(item) or str(item).strip().lower() in ["", "nan", "none", "null"]:
        return False 
    
    # 2. If the valid item is completely missing from the dictionary
    if item not in mapping_dict:
        return True
        
    # 3. If the item is in the dictionary, but the mapped value is invalid/empty
    val = mapping_dict[item]
    if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none", "null"]:
        return True
        
    return False
    
    
def interpolate(
    df: pd.DataFrame, 
    n_points: int, 
    interpolate_fn = lambda x, y: interp1d(x, y, kind='linear')
):
    df = bio.Dataset.convert_from_scientific_notation(df, column_name="CAPACITY")
    df['CONCENTRATION'] = pd.to_numeric(df['CONCENTRATION'])
    df['CAPACITY'] = pd.to_numeric(df['CAPACITY'])
    core_cols = ['POLYMER_USED', 'DRUG', 'CONCENTRATION', 'CAPACITY', 'SOURCE']
    static_cols = [col for col in df.columns if col not in core_cols]
    
    # Group by unique identifiers
    groups = df.groupby(['POLYMER_USED', 'DRUG'])
    interpolated_list = []
    for (polymer, drug), group in groups:
        if len(group) < 2: continue
        logger.debug(f"polymer: {polymer}")
        logger.debug(f"drug: {drug}")
        group = group.sort_values('CONCENTRATION')
        x = group['CONCENTRATION'].values
        y = group['CAPACITY'].values
        f = interpolate_fn(x, y)
        middle_points = get_middle_points(x, n_points)
        middle_results = [f(p) for p in middle_points]
        significant_digits = 5
        middle_points = [round(float(p), significant_digits) for p in middle_points]
        middle_results = [round(float(r), significant_digits) for r in middle_results]
        logger.debug(f"x: {x}")
        logger.debug(f"middle_points: {middle_points}")
        logger.debug(f"middle_results: {middle_results}")
        interp_data = {
            'POLYMER_USED': polymer, 
            'DRUG': drug,
            'CONCENTRATION': middle_points, 
            'CAPACITY': middle_results,
            'SOURCE': 'interpolated'
        }
        for col in static_cols: interp_data[col] = group.iloc[0][col]
        interp_df = pd.DataFrame(interp_data)
        df = pd.concat([df, interp_df], ignore_index=True)
    
    df = df.sort_values(by=['POLYMER_USED', 'DRUG', 'CONCENTRATION'], ascending=True)
    return df

def get_middle_points(vector: np.array, n_middle_points: int) -> np.array:
    logger.trace(f"vector: {vector}")
    vector = np.unique(vector)
    logger.trace(f"vector: {vector}")
    out = []
    for i in range(len(vector)-1):
        logger.trace(f"vector[i+1]: {vector[i+1]}")
        logger.trace(f"vector[i]: {vector[i]}")
        space = np.linspace(vector[i], vector[i+1], num=n_middle_points+2)
        space = space[1:-1]
        out = np.concatenate([out, space])
        logger.trace(f"space: {space}")
    logger.debug(f"out: {out}")
    return np.sort(out)
    

def add_origin_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds CONCENTRATION=0 and CAPACITY=0 for each unique 
    (POLYMER_USED, DRUG, WATER_PH) group.
    """
    new_rows = []
    groups = df.groupby(['POLYMER_USED', 'DRUG', 'WATER_PH'])
    for (polymer, drug, water_ph), group in groups:
        if not (group['CONCENTRATION'] == 0.0).any():
            new_row = group.iloc[0].copy()
            new_row['CONCENTRATION'] = 0.0
            new_row['CAPACITY'] = 0.0
            new_row['SOURCE'] = 'interpolated'
            new_rows.append(new_row)
    if new_rows:
        origin_df = pd.DataFrame(new_rows)
        df = pd.concat([df, origin_df], ignore_index=True)
        logger.debug(f"Added {len(new_rows)} origin (0,0) points.")
    return df


def test_():
    main()
