from cs336_basics.bpe import train_bpe
import time
import pickle
from pathlib import Path

input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
output_dir=Path("outputs")

vocab_size=10000
special_tokens=["<|endoftext|>"]

output_dir.mkdir(parents=True,exist_ok=True)

start_time=time.perf_counter()

vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=vocab_size,
    special_tokens=special_tokens,
)

end_time=time.perf_counter()

vocab_path=output_dir/"tinystories_vocab.pkl"
with open(vocab_path,"wb") as file:
    pickle.dump(vocab,file)

merges_path=output_dir/"tinystories_merges.pkl"
with open(merges_path,"wb") as file:
    pickle.dump(merges,file)

elapsed_time=end_time-start_time

print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Vocab size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

#key需要函数而不是函数执行后的结果
longest_token=max(vocab.values(),key=len)
longest_token_text = longest_token.decode("utf-8", errors="replace")

print(f"Longest token: {longest_token_text}")
print(f"Longest token length: {len(longest_token)} bytes")

print(f"Vocab saved to: {vocab_path}")
print(f"Merges saved to: {merges_path}")