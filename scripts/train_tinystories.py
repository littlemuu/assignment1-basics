
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


def evaluate(
    model,
    val_dataset,
    batch_size,
    context_length,
    device,
    num_eval_batches=20,
):
    was_training = model.training
    model.eval()

    losses = []

    with torch.inference_mode():
        for _ in range(num_eval_batches):

            x, y = get_batch(
                val_dataset,
                batch_size,
                context_length,
                device,
            )

            logits = model(x)
            loss = cross_entropy(logits, y)

            losses.append(loss.item())

    model.train(was_training)

    avg_loss = sum(losses) / len(losses)

    return avg_loss


def main():
    # 1. 随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = "cuda"

    # 2. 训练配置
    vocab_size = 10000
    context_length = 256
    batch_size = 16

    d_model = 256
    num_layers = 4
    num_heads = 4
    d_ff = 688
    theta = 10000.0

    num_steps = 2000
    eval_interval = 200
    log_interval = 100

    max_learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_iters = 100

    weight_decay = 0.01
    max_grad_norm = 1.0

    checkpoint_interval = 500
    resume_path = Path("outputs/checkpoints/tinystories_step1000.pt")

    stop_step = 1001

    # 3. 加载训练数据
    train_path = Path("data/tinystories_train.bin")

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    dataset = np.memmap(
        train_path,
        dtype="<u2",
        mode="r",
    )

    if len(dataset) <= context_length:
        raise ValueError("Dataset is too short")

    val_path = Path("data/tinystories_valid.bin")

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    val_dataset = np.memmap(
        val_path,
        dtype="<u2",
        mode="r",
    )

    if len(val_dataset) <= context_length:
        raise ValueError("Validation dataset is too short")

    print(f"Dataset tokens: {len(dataset):,}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # 4. 初始化 Transformer
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
    ).to(device)

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {num_parameters:,}")

    # 5. 初始化 AdamW
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        weight_decay=weight_decay,
    )

    start_step = 0

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)

        start_step = load_checkpoint(
            resume_path,
            model,
            optimizer,
        )

        print(f"Resumed from step: {start_step}")

    # 6. 开始训练
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    for step in range(start_step + 1, min(num_steps, stop_step) + 1):

        lr = get_lr_cosine_schedule(
            t=step - 1,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=num_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()

        # 随机采样 Batch
        x, y = get_batch(
            dataset,
            batch_size,
            context_length,
            device,
        )

        # Forward
        logits = model(x)

        # Cross Entropy
        loss = cross_entropy(logits, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at step {step}"
            )

        # Backward
        loss.backward()

        # Gradient Clipping
        gradient_clipping(
            model.parameters(),
            max_grad_norm,
        )

        # AdamW 参数更新
        optimizer.step()

        if step == stop_step:
            print(
                f"Resume smoke test | "
                f"Step: {step} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.6e}",
                flush=True,
            )

        # 7. 记录训练日志
        if step == 1 or step % log_interval == 0:
            torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            tokens_seen = (
                (step - start_step)
                * batch_size
                * context_length
            )

            throughput = tokens_seen / elapsed

            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.6e} | "
                f"Tokens/s: {throughput:.1f}",
                flush=True,
            )

        # 定期评估验证集
        if step % eval_interval == 0:
            val_loss = evaluate(
                model=model,
                val_dataset=val_dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                num_eval_batches=20,
            )

            print(
                f"Step {step:03d} | "
                f"Validation Loss: {val_loss:.4f}",
                flush=True,
            )

        if step % checkpoint_interval == 0:
            checkpoint_path = Path(
                f"outputs/checkpoints/tinystories_step{step}.pt"
            )

            checkpoint_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=checkpoint_path,
            )

            print(f"Checkpoint saved: {checkpoint_path}")

    # 8. 统计最终性能
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**2
    )

    print(f"Total training time: {total_time:.2f}s")
    print(f"Peak GPU memory: {peak_memory:.1f} MiB")

if __name__ == "__main__":
    main()
