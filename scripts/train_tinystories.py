
import time
from pathlib import Path
import csv

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

import argparse

def evaluate(
    model,
    val_dataset,
    batch_size,
    context_length,
    device,
    num_eval_batches=20,
):
    was_training = model.training
    rng_state = np.random.get_state()
    losses = []

    try:
        np.random.seed(2026)
        model.eval()

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

    finally:
        np.random.set_state(rng_state)
        model.train(was_training)

    return sum(losses) / len(losses)



def main():

    parser = argparse.ArgumentParser(
        description="Tinystories Learning Rate Sweep Training Script"
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        required=True,
        help="Maximum learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[16, 32, 64],
        required=True,
    )
    args = parser.parse_args()

    if args.max_lr <= 0:
        raise ValueError("Maximum learning rate must be positive")

    # 1. 随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = "cuda"

    # 2. 训练配置
    vocab_size = 10000
    context_length = 256
    batch_size = args.batch_size

    reference_batch_size = 16
    reference_steps = 2000
    eval_batch_size = 16

    token_budget = (
        reference_batch_size
        * reference_steps
        * context_length
    )

    d_model = 256
    num_layers = 4
    num_heads = 4
    d_ff = 688
    theta = 10000.0


    max_learning_rate = args.max_lr
    min_learning_rate = max_learning_rate * 0.1
    run_name = (
        f"batch_sweep_v1_bs{batch_size}"
        f"_lr{max_learning_rate:.0e}"
    )

    weight_decay = 0.01
    max_grad_norm = 1.0

    resume_path = None

    stop_step = num_steps

    tokens_per_step = batch_size * context_length

    assert token_budget % tokens_per_step == 0

    num_steps = token_budget // tokens_per_step
    stop_step = num_steps

    warmup_tokens = 100 * 16 * context_length
    warmup_iters = warmup_tokens // tokens_per_step

    log_interval = (100 * 16 * context_length) // tokens_per_step
    eval_interval = (200 * 16 * context_length) // tokens_per_step
    checkpoint_interval = (500 * 16 * context_length) // tokens_per_step

    assert num_steps * batch_size * context_length == 8_192_000
    
    assert warmup_iters * batch_size * context_length == 409_600

    assert eval_interval * batch_size * context_length == 819_200

    assert resume_path is None

    print(
        f"BS={batch_size} | "
        f"Steps={num_steps} | "
        f"Warmup={warmup_iters} | "
        f"Eval Interval={eval_interval}"
    )

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

    # 初始化实验日志
    end_step = min(num_steps, stop_step)

    log_path = Path(
        f"outputs/experiments/{run_name}.csv"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if start_step == 0:
        file_mode = "x"
        need_header = True
    else:
        if not log_path.exists():
            raise FileNotFoundError(
                f"Missing experiment log: {log_path}"
            )

        file_mode = "a"
        need_header = log_path.stat().st_size == 0

    # 6. 开始训练
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    with log_path.open(
        file_mode, newline="", encoding="utf-8"
    ) as log_file:

        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
                "tokens_seen",
                "train_loss",
                "val_loss",
                "lr",
                "elapsed_seconds",
            ],
        )

        if need_header:
            writer.writeheader()

        for step in range(start_step + 1, end_step + 1):

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

            if step == end_step:
                print(
                    f"Training completed | "
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
            val_loss = None
            if step % eval_interval == 0:
                val_loss = evaluate(
                    model=model,
                    val_dataset=val_dataset,
                    batch_size=eval_batch_size,
                    context_length=context_length,
                    device=device,
                    num_eval_batches=20,
                )

                print(
                    f"Step {step:03d} | "
                    f"Validation Loss: {val_loss:.4f}",
                    flush=True,
                )

            should_log = (
                step == 1
                or step % log_interval == 0
                or val_loss is not None
                or step == end_step
            )

            if should_log:
                torch.cuda.synchronize()

                writer.writerow({
                    "step": step,
                    "tokens_seen": step * x.numel(),
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    'elapsed_seconds': time.perf_counter() - start_time,
                })

                log_file.flush()

            if step % checkpoint_interval == 0:
                checkpoint_path = Path(
                    f"outputs/checkpoints/{run_name}_step{step}.pt"
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
    print(f"Token Budget: {token_budget:,}")


if __name__ == "__main__":
    main()
