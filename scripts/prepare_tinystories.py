import pickle
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer

output_dir = Path("outputs")

with open(output_dir / "tinystories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open(output_dir / "tinystories_merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(
    vocab=vocab,
    merges=merges,
    special_tokens=["<|endoftext|>"],
)

sample = "Once upon a time, there was a little dragon."

token_ids = tokenizer.encode(sample)

decoded = tokenizer.decode(token_ids)

print("Vocab size:", len(tokenizer.vocab))
print("Token IDs:", token_ids)
print("Decoded:", decoded)

assert decoded == sample
