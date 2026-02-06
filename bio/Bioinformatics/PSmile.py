import re
from rdkit import Chem, rdBase
from dataclasses import dataclass
import lele, bio
print(lele)
print(dir(lele))
Smile = lele.type(bio.Bioinformatics.Smile)

@dataclass
class PSmile:
    prechain: str
    chain: str
    pstchain: str
    
    def from_str(psmile: str) -> 'PSmile':
        return bio.Bioinformatics.PSmileMethod.from_str(psmile)
    
    def to_smile(self, n: int) -> Smile:
        return Smile(self.prechain + (n * self.chain) + self.pstchain)

    @staticmethod
    def is_valid(psmile: str) -> bool:
        return bio.Bioinformatics.PSmileMethod.from_str(psmile)
