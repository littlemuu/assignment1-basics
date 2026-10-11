import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def read_chunks(file, max_chars=None):
    remaining = max_chars

    while remaining is None or remaining > 0:
        size = 1024 * 1024
        if remaining is not None:
            size = min(size, remaining)

        chunk = file.read(size)
        if not chunk:
            break

        yield chunk

        if remaining is not None:
            remaining -= len(chunk)


def tokenize_file(tokenizer, input_path, output_path, max_chars=None):
    buffer = []
    total_tokens = 0
    start = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".bin.tmp")

    with input_path.open("r", encoding="utf-8") as source, \
         temp_path.open("wb") as target:

        for token_id in tokenizer.encode_iterable(
            read_chunks(source, max_chars)
        ):
            buffer.append(token_id)

            if len(buffer) >= 100_000:
                np.asarray(buffer, dtype="<u2").tofile(target)
                total_tokens += len(buffer)
                buffer.clear()

                if total_tokens % 5_000_000 == 0:
                    print(f"Encoded {total_tokens:,} tokens", flush=True)

        if buffer:
            np.asarray(buffer, dtype="<u2").tofile(target)
            total_tokens += len(buffer)

    temp_path.replace(output_path)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()

    with open("outputs/tinystories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("outputs/tinystories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    if max(vocab) >= 65536:
        raise ValueError("Token ID exceeds uint16 range")

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"],
    )

    tokenize_file(
        tokenizer,
        args.input_path,
        args.output_path,
        args.max_chars,
    )


if __name__ == "__main__":
    main()