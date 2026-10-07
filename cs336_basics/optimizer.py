import torch
import math

class AdamW(torch.optim.Optimizer):
    
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = {
            "lr":lr,
            "weight_decay":weight_decay,
            "betas":betas,
            "eps":eps,
        }

        super().__init__(params, defaults)


    def step(self, closure=None):
        
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"] 
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state["t"] + 1
                m = state["m"]
                v = state["v"]
                grad = p.grad
                
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1-beta1**t)

                p.data -= weight_decay*lr*p.data
                p.data -= alpha_t*m/(torch.sqrt(v)+eps)
                state["t"] = t
                state["m"] = m
                state["v"] = v


def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):

    if t < warmup_iters:
        return t / warmup_iters * max_learning_rate

    elif t<= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = math.cos(math.pi * progress)
        return min_learning_rate + 0.5 * (1 + cosine) * (max_learning_rate - min_learning_rate)

    else:
        return min_learning_rate