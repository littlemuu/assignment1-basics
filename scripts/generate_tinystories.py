import torch
import pickle
from pathlib import Path

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


@torch.inference_mode()
def generate(
    model,
    input_ids,
    max_new_tokens,
    context_length=256,
    temperature=1.0,
    top_p=0.9,
    eos_token_id=None,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    model.eval()

    for _ in range(max_new_tokens):
        # 1. 截取上下文
        context = input_ids[:, -context_length:]

        # 2. Forward，提取最后位置
        logits = model(context)[:, -1, :]

        # 3. Temperature
        probs = torch.softmax(
            logits / temperature,
            dim=-1,
        )

        if top_p == 1.0:
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

        else:
            # 4. 按概率降序排列
            sorted_probs, sorted_indices = torch.sort(
                probs,
                dim=-1,
                descending=True,
            )

            # 5. 累计概率
            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1,
            )

            # 6. Top-p Mask
            remove_mask = cumulative_probs >= top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False

            # 7. 过滤 + 重新归一化
            filtered_probs = sorted_probs.masked_fill(
                remove_mask, 0.0
            )
            filtered_probs = filtered_probs / filtered_probs.sum(
                dim=-1,
                keepdim=True,
            )

            # 8. Sampling
            sampled_index = torch.multinomial(
                filtered_probs,
                num_samples=1,
            )

            # 9. 恢复原始 Token ID
            next_token = torch.gather(
                sorted_indices,
                dim=-1,
                index=sampled_index,
            )

        # 10. EOS 停止
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # 11. 拼接 Token
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=-1,
        )

    return input_ids


def main():
    # 1. 基本配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_size = 10000
    context_length = 256

    d_model = 256
    num_layers = 4
    num_heads = 4
    d_ff = 688
    theta = 10000.0

    max_new_tokens = 100

    eos_ids = tokenizer.encode("<|endoftext|>")
    assert len(eos_ids) == 1

    eos_token_id = eos_ids[0]

    prompt = "Once upon a time, there was a little cat"

    # 2. 加载 BPE Tokenizer
    with open("outputs/tinystories_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("outputs/tinystories_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, special_tokens=["<endoftext>"])

    # 3. 初始化 TransformerLM
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
    )

    # 4. 加载训练好的 Checkpoint
    checkpoint_path = Path(
        "outputs/checkpoints/tinystories_step2000.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint step: {checkpoint['iteration']}")

    # 5. 将 Prompt 转换成 Token IDs
    token_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # 6. 调用 generate()
    for temperature in [0.7, 1.2, 1.5]:
        
        output_ids = generate(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            temperature=temperature,
            top_p=0.9,
            eos_token_id=eos_token_id
        )

        # 7. 解码生成结果
        generated_text = tokenizer.decode(output_ids[0].cpu().tolist())

        print(f"\n=== Temperature: {temperature} ===")
        print(generated_text)
    

if __name__ == "__main__":
    main()