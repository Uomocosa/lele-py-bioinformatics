import lele
from bio.Bioinformatics import SmileMethod
from typing import Optional

class Smile(str): 
    """Represents a single SMILES string"""
        
    @property
    def is_valid(self) -> bool:
        """Checks if the SMILES string is valid according to RDKit."""
        return SmileMethod.is_smiles_string_valid(self)
    
    # @property
    def canonicalize(self) -> Optional['Smile']:
        """Canonicalizes the SMILES string using RDKit."""
        return Smile(SmileMethod.canonicalize_smiles(self))


def test():
    import lele
    smile = Smile("COC")
    assert lele.isinstance(smile, Smile)
    assert lele.isinstance(smile.canonicalize(), Smile)
