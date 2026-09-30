import torch.nn as nn
import torch
import math

class Linear(nn.Module):

    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        shape = (d_out, d_in)
        tensor = torch.empty(shape, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (d_in + d_out))
        self.weight = nn.Parameter(tensor)
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)


    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        shape = (num_embeddings, embedding_dim)
        tensor = torch.empty(shape, device=device, dtype=dtype)
        self.weight = nn.Parameter(tensor)
        torch.nn.init.trunc_normal_(
            self.weight,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model=d_model
        self.eps=eps

        # torch.ones创建一个全为1的张量
        tensor=torch.ones(d_model, device=device, dtype=dtype)
        self.gamma=nn.Parameter(tensor)


    def forward(self, x:torch.Tensor):

        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True)+self.eps
            )
        result = x/rms*self.gamma
        return result.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def silu(self, x:torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x:torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta:float, d_k:int, max_seq_len:int, device=None):
        super().__init__()

        self.d_k = d_k
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RotaryPositionalEmbedding.")

        P = d_k // 2
        token_positions = torch.arange(max_seq_len, device=device).unsqueeze(1)
        pair_id = torch.arange(P, device=device, dtype=torch.float32)
        # 每个pair的频率
        exponent = -2 * pair_id / d_k
        freq = theta ** exponent
        angles = token_positions * freq

        # register_buffer(name, tensor, persistent=True) 的意思是：
        # 把一个 Tensor 注册成 nn.Module 的 buffer。
        # 它不会像 nn.Parameter 那样被优化器训练，
        # 但会跟随模块一起 .to(device) 移动。
        # persistent=False 则表示不把它写进 state_dict；
        # RoPE 的 sin/cos 可以随时重新计算，所以很适合这么做。
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            token_positions: torch.Tensor
    )-> torch.Tensor:
        
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        paired = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        first = paired[..., 0]
        second = paired[..., 1]

        rotated_first = cos*first - sin*second
        rotated_second = cos*second + sin*first
        result = torch.stack([rotated_first, rotated_second], dim=-1)
        result = result.reshape(x.shape)
        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    max_value = torch.max(x, dim=dim, keepdim=True).values
    shifted_x = x - max_value

    # exp_x.shape=(...,n),dim=-1
    # sum_exp.shape=(...,1)
    exp_x = torch.exp(shifted_x)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp


def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor|None = None,
) -> torch.Tensor:

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)

    return attention_weights @ V


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float|None = None,
        max_seq_len: int|None =None,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = None


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor|None =None,
    ) -> torch.Tensor:

        # split heads
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(*Q.shape[:-1], self.num_heads, self.d_k)
        Q = Q.transpose(-3, -2)
        K = K.reshape(*K.shape[:-1], self.num_heads, self.d_k)
        K = K.transpose(-3, -2)
        V = V.reshape(*V.shape[:-1], self.num_heads, self.d_k)
        V = V.transpose(-3, -2)

        # apply RoPE to Q/K
        if self.rope is not None:
            assert token_positions is not None
            Q = self.rope(Q, token_positions.unsqueeze(-2))
            K = self.rope(K, token_positions.unsqueeze(-2))

        # build causal mask
        seq_len = x.shape[-2]
        # torch.tril:保留下三角部分
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        # scaled dot-product attention
        attention_output = scaled_dot_product_attention(Q, K, V, mask)

        # merge heads
        output = attention_output.transpose(-3, -2)
        output = output.reshape(*output.shape[:-2], self.d_model)

        # output projection
        output = self.output_proj(output)

        return output


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
    ):
        super().__init__()

        self.norm1 = RMSNorm(d_model)

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )

        self.norm2 = RMSNorm(d_model)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:

        y = x + self.attention(self.norm1(x), token_positions)
        z = y + self.ffn(self.norm2(y))

        return z


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        self.lm_head = Linear(d_model,vocab_size)


    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        
        x = self.token_embeddings(in_indices)

        seq_len = in_indices.shape[-1]

        token_positions = torch.arange(
            seq_len,
            device=in_indices.device,
        )

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits