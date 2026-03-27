from dataclasses import dataclass
from typing import Callable
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import bio
from loguru import logger


METHOD_DICT = {
    "interpolate": lambda df, n, fn: interpolate(df, n, fn),
    "add_origins": lambda df, n, fn: add_origin_points(df),
    "add_origins_then_interpolate": lambda df, n, fn: interpolate(add_origin_points(df), n, fn),
    "interpolate_then_add_origins": lambda df, n, fn: add_origin_points(interpolate(df, n, fn)),
}

INTERPOLATE_FN_DICT = {
    "linear": lambda x, y: interp1d(x, y, kind='linear'),
}

@dataclass
class Options:
    method: str = "interpolate_then_add_origins"
    interpolate_fn: str = "linear"
    n_points: int = 2


from bio.__global__ import CACHE_MEMORY
@CACHE_MEMORY.cache
def increment_dataset(
    df: pd.DataFrame,
    options: Options = Options()
):
    assert options.method in METHOD_DICT, f"Unknown method: {options.method}, use one of {list(METHOD_DICT.keys())}"
    assert options.interpolate_fn in INTERPOLATE_FN_DICT, f"Unknown interpolate function: {options.interpolate_fn}, use one of {list(INTERPOLATE_FN_DICT.keys())}"
    method = METHOD_DICT[options.method]
    interpolate_fn = INTERPOLATE_FN_DICT[options.interpolate_fn]
    return method(df, options.n_points, interpolate_fn)


def interpolate(df: pd.DataFrame, n_points: int, interpolate_fn):
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



def test_interpolate():
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    df = pd.read_csv(PDCC_CSV)
    len_before = len(df)
    df = increment_dataset(df, Options(method="interpolate"))
    logger.info(f"Interpolated: gained {len(df) - len_before} data.")
    assert len(df) > len_before


def test_add_origins():
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    df = pd.read_csv(PDCC_CSV)
    len_before = len(df)
    df = increment_dataset(df, Options(method="add_origins"))
    logger.info(f"Added origin points: gained {len(df) - len_before} data.")
    assert len(df) > len_before


def test_interpolate_then_add_origins():
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    df = pd.read_csv(PDCC_CSV)
    len_before = len(df)
    df = increment_dataset(df, Options(method="interpolate_then_add_origins"))
    logger.info(f"Interpolated and then added origin points: gained {len(df) - len_before} data.")
    assert len(df) > len_before
    
    
def test_add_origins_then_interpolate():
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    df = pd.read_csv(PDCC_CSV)
    len_before = len(df)
    df = increment_dataset(df, Options(method="add_origins_then_interpolate"))
    logger.info(f"Added origin points and then interpolated: gained {len(df) - len_before} data.")
    assert len(df) > len_before
