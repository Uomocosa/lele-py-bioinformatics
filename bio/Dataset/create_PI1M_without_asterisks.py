from loguru import logger

def create_PI1M_without_asterisks(input_file, output_file):
    """STUPID IDEA DO NOT USE THIS"""
    logger.warning("STUPID IDEA DO NOT USE THIS")
    assert input_file.exists()
    psmiles = input_file.read_text().split("\n")
    assert len(psmiles) > 0
    smiles = [ps.removeprefix('*').removesuffix('*') for ps in psmiles]
    output_file.touch(exist_ok=True)
    output_file.write_text("\n".join(smiles))


import pytest
@pytest.mark.skip(reason="STUPID IDEA DO NOT USE THIS")
def test_():
    from bio.Dataset.__global__ import HELPER_DIR, PI1M, SMILES_VOCABULARY
    PI1M_without_asterisks = HELPER_DIR / "PI1M_without_asterisks.csv"
    create_PI1M_without_asterisks(
        input_file = PI1M, 
        output_file = PI1M_without_asterisks, 
    )
