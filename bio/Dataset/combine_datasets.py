import bio

import csv
from pathlib import Path
from typing import List, Iterable, Any
import heapq
from itertools import chain


def combine_datasets(csv_files_input: List[Path], output_path: Path):
    handles = [open(f, 'r') for f in csv_files_input]
    # Skip first line of each file, consider it the header
    for h in handles: next(h) 

    # zip(*handles) takes one row from each file at a time
    # chain.from_iterable flattens those tuples into a single stream
    combined_dataset = chain.from_iterable(zip(*handles))
    with open(output_path, 'w') as file:
        file.write("PSMILES\n"+"".join(combined_dataset))
        
        
import pytest
@pytest.mark.above10s
def test_():
    from bio.Dataset.__global__ import PI1M, ZINC_BASE_CSV, COMBINED_PI1M_ZINC
    combine_datasets(
        csv_files_input = [PI1M, ZINC_BASE_CSV], 
        output_path = COMBINED_PI1M_ZINC,
    )
