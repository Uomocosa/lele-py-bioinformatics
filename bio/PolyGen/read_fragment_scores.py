import os
import json
from bio.PolyGen.__global__ import FPSCORES, FPSCORES_JSON
import lele

"""
NOTE! Refactored using GEMINI (AI)
NOTE! Useless for my application

FUNCTION SUMMARY:
Loads a pre-computed dictionary of fragment importance scores from a JSON file.
The function flattens a nested list structure into a lookup table where:
- Keys: Fragment identifiers (strings or integers).
- Values: The statistical score/weight associated with that fragment.
This is typically used to calculate the Synthetic Accessibility (SA) of a molecule.
"""
def read_fragment_scores(file_path=FPSCORES_JSON):
    """
    NOTE! Useless for my application
    """
    global FPSCORES
    if FPSCORES is not None: return FPSCORES
    
    file_path = lele.P(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find fragment scores file at {file_path}")

    with open(file_path, 'r') as f: data = json.load(f)

    # The JSON structure is expected to be: [[score, frag1, frag2, ...], [...]]
    # We transform this into: {frag1: score, frag2: score, ...}
    out_dict = {}
    for entry in data:
        score = float(entry[0])
        for fragment_id in entry[1:]:
            out_dict[fragment_id] = score
            
    FPSCORES = out_dict
    return FPSCORES


def test_():
    from .__global__ import HELPER_DIR, FPSCORES
    global FPSCORES
    FPSCORES = None # FOR THIS TEST WE NEED IT TO BE UNLOADED
    # 1. Setup: Create a temporary dummy JSON for testing
    test_filename = HELPER_DIR/"test_fpscores.json"
    dummy_data = [
        [0.5, "fragment_A", "fragment_B"],
        [-1.2, "fragment_C"]
    ]
    
    with open(test_filename, "w") as f:
        json.dump(dummy_data, f)

    try:
        FPSCORES = read_fragment_scores(test_filename)
        
        # 3. Assertions
        print("Loaded Scores:", FPSCORES)
        assert FPSCORES["fragment_A"] == 0.5
        assert FPSCORES["fragment_B"] == 0.5
        assert FPSCORES["fragment_C"] == -1.2
        assert FPSCORES is not None
        
    finally:
        # Cleanup
        if os.path.exists(test_filename):
            os.remove(test_filename)
            FPSCORES = None # UNLOAD DUMMY DATA
