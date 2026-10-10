from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.checkpoint import load_checkpoint

import numpy as np

from scripts.train import train_step, evaluate
from cs336_basics.optimizer import get_lr_cosine_schedule

device = "cpu"

model = TransformerLM(
    vocab_size=32,
    context_length=16,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
    theta=10000.0,
).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)

last_step = load_checkpoint(
    src="outputs/checkpoints/smoke_step100.pt",
    model=model,
    optimizer=optimizer,
)

print(f"Restored at step: {last_step}")

dataset = np.tile(
    np.arange(32, dtype=np.int64),
    64,
)

batch_size = 8
context_length = 16
additional_steps = 10

for step in range(last_step + 1, last_step + additional_steps + 1):

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

    print(
        f"Step {step:03d} | "
        f"Loss: {loss_value:.4f} | "
        f"LR: {lr:.6f}"
    )

from pathlib import Path
from cs336_basics.checkpoint import save_checkpoint

new_step = last_step + additional_steps

checkpoint_path = Path(
    f"outputs/checkpoints/smoke_step{new_step}.pt"
)

checkpoint_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

save_checkpoint(
    model,
    optimizer,
    iteration=new_step,
    out=checkpoint_path,
)

print(f"Saved checkpoint at step {new_step}")

val_dataset = np.tile(
    np.arange(32, dtype=np.int64),
    8,
)

val_loss = evaluate(
    model,
    val_dataset,
    batch_size=8,
    context_length=16,
    device=device,
)

print(f"Validation Loss: {val_loss:.4f}")