import bio
import torch
from deepchem.feat.smiles_tokenizer import SmilesTokenizer

class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        parent: bio.Dataset.LabeledSmiles, 
        tokenizer: SmilesTokenizer, 
        block_size: int
    ):
        self.parent = parent
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Assuming based on common simple tokenizers:
        self.pad_id = 0 
        self.eos_id = tokenizer.vocab_size - 1 # Or however your tokenizer defines it

    def __len__(self): 
        return len(self.parent)

    def __getitem__(self, idx):
        smile, label = self.parent[idx]
        tokens = self.tokenizer.encode(smile)
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        if len(tokens) < self.block_size:
            tokens = tokens + [self.pad_id] * (self.block_size - len(tokens))
            
        x = torch.tensor(tokens, dtype=torch.long)
        y = label
        return x, y
    
    # @staticmethod
    # def features():
    #     return [label for smile, label in self.parent]
        
    # @staticmethod
    # def smiles():
    #     return [smile for smile, label in self.parent]

def test_usage():
    from bio.Dataset.__global__ import ZINC_BASE_CSV
    BLOCK_SIZE = 8
    smiles = bio.Dataset.UnlabeledSmiles.load(ZINC_BASE_CSV, max_dataset_size=100)
    dataset = smiles.to_torch_dataset(BLOCK_SIZE)
    print(f"dataset: {dataset}")
    for i in range(0,100): print(f"dataset[{i}]: {dataset[i]}")


def test_internal():
    from bio.Dataset.__global__ import ZINC_ZSCORE_CSV, SMILES_VOCABULARY
    BLOCK_SIZE = 8
    smiles = bio.Dataset.LabeledSmiles.load(ZINC_ZSCORE_CSV, max_dataset_size=100)
    tokenizer = SmilesTokenizer(SMILES_VOCABULARY)
    dataset = TorchDataset(smiles, tokenizer, BLOCK_SIZE)
    print(f"dataset: {dataset}")
    for i in range(0,100): print(f"dataset[{i}]: {dataset[i]}")
