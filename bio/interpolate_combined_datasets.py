import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import bio
from loguru import logger
from bio.__global__ import DATASETS_DIR
COMBINED_DATASETS = DATASETS_DIR / "COMBINED_DATASETS" / "polymer_drug_concentration_capacity.csv"
INTERPOLATED_CSV = DATASETS_DIR / "COMBINED_DATASETS" / "interpolated.csv"
N_POINTS_TO_INTERPOLATE = 1

def main():
    assert COMBINED_DATASETS.exists()
    df = pd.read_csv(COMBINED_DATASETS)
    df = bio.Dataset.convert_from_scientific_notation(df, column_name="CAPACITY")
    df['CONCENTRATION'] = pd.to_numeric(df['CONCENTRATION'])
    df['CAPACITY'] = pd.to_numeric(df['CAPACITY'])
    
    # Group by unique identifiers
    groups = df.groupby(['POLYMER_USED', 'DRUG'])
    interpolated_list = []
    n_points = N_POINTS_TO_INTERPOLATE
    for (polymer, drug), group in groups:
        if len(group) < 2: continue
        
        logger.debug(f"polymer: {polymer}")
        logger.debug(f"drug: {drug}")
        logger.debug(f"group: {group}")
        
        group = group.sort_values('CONCENTRATION')
        x = group['CONCENTRATION'].values
        y = group['CAPACITY'].values
        f = interp1d(x, y, kind='linear')
        # middle_points = [np.linspace(x[i], x[i+1], num=N_POINTS_TO_INTERPOLATE + 2)[1:-1] for i in range(len(x)-1)]
        middle_points = get_middle_points(x, N_POINTS_TO_INTERPOLATE)
        middle_results = [f(p) for p in middle_points]
        significant_digits = 5
        middle_points = [round(float(p), significant_digits) for p in middle_points]
        middle_results = [round(float(r), significant_digits) for r in middle_results]
        logger.debug(f"x: {x}")
        logger.debug(f"middle_points: {middle_points}")
        logger.debug(f"middle_results: {middle_results}")
        interp_df = pd.DataFrame({
            'POLYMER_USED': polymer, 
            'DRUG': drug, 
            'CONCENTRATION': middle_points, 
            'CAPACITY': middle_results
        })
        df = pd.concat([df, interp_df], ignore_index=True)
    
    df = df.sort_values(by=['POLYMER_USED', 'DRUG', 'CONCENTRATION'], ascending=True)
    print(df)
    df.to_csv(INTERPOLATED_CSV, index=False)

def get_middle_points(vector: np.array, n_middle_points: int) -> np.array:
    print(f"vector: {vector}")
    vector = np.unique(vector)
    logger.debug(f"vector: {vector}")
    out = []
    for i in range(len(vector)-1):
        logger.debug(f"vector[i+1]: {vector[i+1]}")
        logger.debug(f"vector[i]: {vector[i]}")
        space = np.linspace(vector[i], vector[i+1], num=n_middle_points+2)
        space = space[1:-1]
        out = np.concatenate([out, space])
        logger.debug(f"space: {space}")
    logger.debug(f"out: {out}")
    return np.sort(out)
    

def test_():
    main()
