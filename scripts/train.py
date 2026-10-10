import torch
import numpy as np

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint


def train_step(
    model,
    optimizer,
    dataset,
    batch_size,
    context_length,
    device,
):
    model.train()

    optimizer.zero_grad()

    x, y = get_batch(
        dataset,
        batch_size,
        context_length,
        device,
    )

    logits = model(x)

    loss = cross_entropy(logits, y)

    loss.backward()

    parameters = model.parameters()
    gradient_clipping(parameters, 1.0)

    optimizer.step()

    return loss.item()


def evaluate(
    model,
    val_dataset,
    batch_size,
    context_length,
    device,
):
    model.eval()

    with torch.no_grad():

        x, y = get_batch(
            val_dataset,
            batch_size,
            context_length,
            device,
        )

        logits = model(x)

        val_loss = cross_entropy(logits, y)

    model.train()

    return val_loss.item()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    device = "cpu"
    vocab_size = 32
    context_length = 16
    batch_size = 8

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        theta=10000.0,
    ).to(device)

    dataset = np.tile(
        np.arange(vocab_size, dtype=np.int64),
        64,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )

    x, y = get_batch(
        dataset,
        batch_size,
        context_length,
        device,
    )

    with torch.no_grad():
        logits = model(x)

    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    print("Logits shape:", logits.shape)

    num_iters = 100

    for step in range(1, num_iters + 1):

        lr = get_lr_cosine_schedule(
            t=step,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=10,
            cosine_cycle_iters=100,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        loss_value = train_step(
            model,
            optimizer,
            dataset,
            batch_size,
            context_length,
            device,
        )

        if step == 1 or step % 10 == 0:
            print(
                f"Step {step:03d} | "
                f"Loss: {loss_value:.4f} | "
                f"LR: {lr:.6f}"
            )

    from pathlib import Path

    checkpoint_path = Path("outputs/checkpoints/smoke_step100.pt")

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_checkpoint(
        model,
        optimizer,
        iteration=num_iters,
        out=checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")