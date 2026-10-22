
import csv
from pathlib import Path

import matplotlib.pyplot as plt

log_dir = Path("outputs/experiments")

learning_rates = [
    1e-4,
    3e-4,
    1e-3,
    2e-3,
    3e-3,
]

plt.figure(figsize=(9, 5))

for max_lr in learning_rates:
    run_name = f"lr_{max_lr:.0e}"
    path = log_dir / f"{run_name}.csv"

    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row["val_loss"]
        ]

    if not rows:
        raise ValueError(f"No validation results: {path}")

    tokens = [
        int(row["tokens_seen"]) / 1e6
        for row in rows
    ]

    losses = [
        float(row["val_loss"])
        for row in rows
    ]

    plt.plot(
        tokens,
        losses,
        marker="o",
        label=f"Max LR = {max_lr:.0e}",
    )

    print(
        f"{run_name} | "
        f"Final Loss: {losses[-1]:.4f} | "
        f"Best Loss: {min(losses):.4f}"
    )

plt.xlabel("Training Tokens Seen (Millions)")
plt.ylabel("Validation Cross-Entropy Loss")
plt.title("TinyStories Learning Rate Sweep")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

output = log_dir / "lr_sweep_comparison.png"
plt.savefig(output, dpi=160)
plt.close()

print(f"Saved: {output}")
