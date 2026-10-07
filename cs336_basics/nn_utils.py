import torch


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    max_values = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs-max_values

    exp_inputs = torch.exp(shifted_inputs)
    sum_exp = torch.sum(exp_inputs, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    target_indices = targets.unsqueeze(-1)
    correct_logits = torch.gather(
        shifted_inputs,
        dim=-1,
        index=target_indices,
    )

    losses = log_sum_exp-correct_logits

    return losses.mean()


def gradient_clipping(
    parameters,
    max_l2_norm: float,
) -> None:
    parameters = list(parameters)
    total_squared_norm = 0.0

    for p in parameters:
        if p.grad is None:
            continue

        grad = p.grad
        total_squared_norm += (grad ** 2).sum()

    total_norm = torch.sqrt(total_squared_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for p in parameters:
            if p.grad is None:
                continue

            # p.grad = p.grad * scale,这样写不对，要原地修改
            p.grad.mul_(scale)