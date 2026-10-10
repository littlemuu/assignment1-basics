from pathlib import Path
import pickle
import time

from cs336_basics.bpe import train_bpe


def main():
    input_path = Path("data/tinystories_64m.txt")
    output_dir = Path("outputs/bpe_64m")

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start

    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    print(f"Training time: {elapsed:.2f}s")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()