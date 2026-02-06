from dataclasses import dataclass, field
from typing import List, Dict # Using specific types is good practice


@dataclass
class Vocabolary():
    """NOT SURE IF THIS IS USEFUL!"""
    tokens: List[str]
    special_tokens: List[str]
    max_seq_length: int
    token_to_int: Dict[str, int] = field(init=False)
    int_to_token: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.token_to_int = {token: i for i, token in enumerate(self.tokens)}
        self.int_to_token = {i: token for i, token in enumerate(self.tokens)}

    @property
    def vocab_size(self):
        return len(self.tokens)
	

def test_():
    tokens = ["<sos>", "<eos>", "a", "bc", "d", "ef"]
    special_tokens = ["<sos>", "<eos>"]
    max_seq_length = 10
    vocabolary = Vocabolary(tokens, special_tokens, max_seq_length)
    print(vocabolary)
    print(vocabolary.vocab_size)
