import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:

    N = len(dataset)

    starts = np.random.randint(0, N - context_length, size=batch_size)

    inputs = [
        dataset[start: start + context_length]
        for start in starts
    ]

    targets = [
        dataset[start + 1: start + context_length + 1]
        for start in starts
    ]

    x = torch.as_tensor(
        np.stack(inputs),
        dtype=torch.long,
        device=device
    )

    y = torch.as_tensor(
        np.stack(targets),
        dtype=torch.long,
        device=device
    )

    return x,y