import bio

import csv
from pathlib import Path
from typing import List, Iterable, Any
import heapq

def combine_datasets(csv_files_input: List[Path], output_path: Path):
    # 1. Get lengths to calculate proportions
    lengths = [get_line_count(p) for p in csv_files_input]
    
    # 2. Open all files (using ExitStack to handle N files safely)
    from contextlib import ExitStack
    
    with ExitStack() as stack:
        files = [stack.enter_context(open(p, 'r', encoding='utf-8')) for p in csv_files_input]
        readers = [csv.DictReader(f) for f in files]
        
        # 3. Use our generator to mix them
        mixed_data = proportional_interleave(readers, lengths)
        
        # 4. Write output
        with open(output_path, 'w', encoding='utf-8', newline='') as fout:
            writer = None
            for row in mixed_data:
                if not writer:
                    writer = csv.DictWriter(fout, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                
def get_line_count(file_path: Path) -> int:
    """Quickly count lines in a CSV (subtracting 1 for header)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1 

def proportional_interleave(datasets: List[Iterable[Any]], lengths: List[int]):
    """
    Interleaves N datasets based on their proportional lengths.
    """
    total_elements = sum(lengths)
    # We use a priority queue to always pick the dataset that is 
    # 'most behind' its ideal distribution schedule.
    # Heap stores: (next_scheduled_time, dataset_index)
    pq = []
    
    for i, length in enumerate(lengths):
        if length > 0:
            # Stride is how 'far' we jump after each pick. 
            # Larger datasets have smaller strides.
            stride = total_elements / length
            # Start at stride/2 to center the distribution (Bresenham-style)
            heapq.heappush(pq, (stride / 2, i, stride))

    # Convert datasets to iterators
    iterators = [iter(d) for d in datasets]

    while pq:
        next_time, idx, stride = heapq.heappop(pq)
        
        try:
            yield next(iterators[idx])
            # Schedule the next appearance for this dataset
            heapq.heappush(pq, (next_time + stride, idx, stride))
        except StopIteration:
            # Dataset exhausted earlier than expected
            pass


import pytest
@pytest.mark.above10s
def test_():
    from bio.Datasets.__global__ import PI1M, ZINC_BASE, COMBINED_PI1M_ZINC
    combine_datasets(
        csv_files_input=[PI1M, ZINC_BASE], 
        output_file=COMBINED_PI1M_ZINC
    )
