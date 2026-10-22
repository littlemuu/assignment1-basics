
import csv
from pathlib import Path

import matplotlib.pyplot as plt

log_path = Path(
    "outputs/experiments/tinystories_smoke_v1.csv"
)

output_path = Path(
    "outputs/experiments/tinystories_smoke_loss.png"
)

train_tokens = []
train_losses = []

val_tokens = []
val_losses = []

# 1. 读取 CSV
with log_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        tokens = int(row["tokens_seen"])

        train_tokens.append(tokens)
        train_losses.append(float(row["train_loss"]))

        if row["val_loss"].strip():
            val_tokens.append(tokens)
            val_losses.append(float(row["val_loss"]))

# 2. 绘制曲线
plt.figure(figsize=(8, 5))

plt.plot(
    train_tokens,
    train_losses,
    marker="o",
    label="Training Loss",
)

plt.plot(
    val_tokens,
    val_losses,
    marker="s",
    label="Validation Loss",
)

# 3. 图表信息
plt.xlabel("Tokens Seen")
plt.ylabel("Cross-Entropy Loss")
plt.title("TinyStories Training Experiment")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 4. 保存结果
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=160)
plt.close()

print(f"Loss curve saved to: {output_path}")
