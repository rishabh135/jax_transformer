"""
The 2024 Transformer (the Noam Transformer):
- RMSNorm
- GQA or some combination 
- Sliding window attention 
- Swiglu
- RoPE (Rotary Positional Embedding)

LLM Arches:
                hidden | MLP mult.   |  n_layers | rope_theta |  GQA Group Size  | GLU Act. |  ops 

Gemma 2 9B      3584   |     4x      |    42     |   10000    |        2         |  GELU Tanh | norm -> attn -> norm -> add -> norm -> mlp -> norm -> add 
Llama 3 8B      4096   |     3.5x    |    32     |   50000    |        4         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
Mistral 7B      4096   |     3.5x    |    32     |  1000000   |        4         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
Qwen 2 7B       3584   |    ~5.29x   |    28     |  1000000   |        7         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
InternLM2.5 7B  4096   |     3.5x    |    32     |  50000000  |        4         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
DeepSeek 6.7B   4096   |     2.6875x |    32     |  100000    |        1         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 

Phi 3 14B |     4096   |     4.375x  |    40     |   10000    |        4         |  SILU      | norm -> attn -> (drop) add -> norm -> mlp -> (drop) add 

Gemma 2 27|     4608   |     8x      |    46     |   10000    |        2         |  GELU Tanh | norm -> attn -> norm -> add -> norm -> mlp -> norm -> add 
DeepSeek 33B    7168   |     ~2.68x  |    62     |  100000    |        7         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
Llama 3 70B     8192   |     3.5x    |    80     |   50000    |        8         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 
Qwen 2 72B      8192   |    ~3.61x   |    80     |  1000000   |        8         |  SILU      | norm -> attn -> add -> norm -> mlp -> add 

Others:
- Gemma 2 uses logit softcapping (50), query pre attention scaling

References:
- https://github.com/naklecha/llama3-from-scratch/tree/main
- https://github.com/xjdr-alt/simple_transformer
- https://github.com/google/gemma_pytorch/tree/main
- https://github.com/hkproj/pytorch-llama/tree/main
"""

import torch
import torch.nn.functional as F
from typing import List, NamedTuple


NUM_Q_HEADS = 32  # Llama numbers
NUM_KV_HEADS = 8  # Llama numbers
SLIDING_WINDOW_SIZE = 4096


class LayerWeights(NamedTuple):
    input_norm: torch.Tensor  # (hidden)
    post_attn_norm: torch.Tensor  # (hidden)
    q_proj: torch.Tensor  # (hidden, q_intermediate)
    k_proj: torch.Tensor  # (hidden, kv_intermediate)
    v_proj: torch.Tensor  # (hidden, kv_intermediate)
    o_proj: torch.Tensor  # (q_intermediate, hidden)
    gate_proj: torch.Tensor  # (hidden, intermediate)
    up_proj: torch.Tensor  # (hidden, intermediate)
    down_proj: torch.Tensor  # (intermediate, hidden)


class TransformerWeights(NamedTuple):
    layers: List[LayerWeights]
    token_emb: torch.Tensor  # (vocab_size, hidden)
    final_norm: torch.Tensor  # (hidden)
    lm_head: torch.Tensor  # (hidden, vocab_size)


def norm(x: torch.Tensor, weight: torch.Tensor):
    in_dtype = x.dtype
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)  # eps might change depending on the model
    return weight * out.to(in_dtype)


def ffn(x: torch.Tensor, weights: LayerWeights):
    gate = F.silu(x @ weights.gate_proj)
    fused = gate * (x @ weights.up_proj)
    return fused @ weights.down_proj


def rope(x: torch.Tensor, freqs_cis: torch.Tensor):
    def rotate(x):
        """
        rotate_half(torch.arange(4))
        > tensor([-2, -3,  0,  1])
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos, sin = freqs_cis
    cos, sin = cos.type_as(x), sin.type_as(x)
    right = rotate(x.reshape(*x.shape[:-1], -1, 2)).reshape(x.shape)
    out = x * cos + right * sin
    return out.to(x.dtype)


def attn(
    x: torch.Tensor,
    weights: LayerWeights,
    freqs_cis: tuple,
    sliding_window_size=None,
):
    bs, seq_len, d_model = x.shape
    xq, xk, xv = x @ weights.q_proj, x @ weights.k_proj, x @ weights.v_proj
    xq = xq.view(bs, seq_len, NUM_Q_HEADS, -1).transpose(1, 2)  # (bs, NUM_Q_HEADS, seq_len, q_intermediate)
    xk = xk.view(bs, seq_len, NUM_KV_HEADS, -1).transpose(1, 2)  # (bs, NUM_KV_HEADS, seq_len, kv_intermediate)
    xv = xv.view(bs, seq_len, NUM_KV_HEADS, -1).transpose(1, 2)  # (bs, NUM_KV_HEADS, seq_len, kv_intermediate)
    head_dim = xq.shape[-1]

    # Treat GQA as MHA and just repeat along the head dimension
    xk = torch.repeat_interleave(xk, NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    xv = torch.repeat_interleave(xv, NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    xq = rope(xq, freqs_cis)
    xk = rope(xk, freqs_cis)

    attn_scores = (xq @ xk.transpose(2, 3)) * (head_dim**-0.5)
    mask = torch.triu(torch.full((bs, seq_len, seq_len), -2.3819763e38), diagonal=1)  # This number is taken from Gemma
    if sliding_window_size is not None:  # Sliding window attention
        all_ones = torch.ones((seq_len, seq_len))
        sliding_mask = torch.triu(all_ones, -1 * sliding_window_size + 1) * torch.tril(all_ones, sliding_window_size - 1)
        mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
    mask = mask.to(x.device, x.dtype)
    attn_scores = attn_scores + mask
    attn_probs = F.softmax(attn_scores, dim=-1)
    attn_out = attn_probs @ xv
    attn_out = attn_out.transpose(1, 2).contiguous().view(bs, seq_len, -1)
    return attn_out @ weights.o_proj


# for efficiency, should precompute for 0..max_length * 2 then select [:curr_length]
def precompute_freqs_cis(head_dim: int, seq_len: int, base_theta: float = 500000.0):
    inv_freqs = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # Eq 15: theta_{1} ... theta_{dim/2}. Shape: (dim/2)
    m = torch.arange(seq_len)  # all possible position indices
    freqs = torch.outer(m, inv_freqs).float()  # [m_i * theta_j] for all i (positions) and j (frequencies). Shape: (seq_len, dim/2) | freqs[i][j] == m[i] * inv_freqs[j]
    cos = torch.cos(freqs)  # Shape: (seq_len, dim/2)
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # Shape: (seq_len, dim)
    sin = torch.sin(freqs)  # Shape: (seq_len, dim/2)
    sin = torch.repeat_interleave(sin, 2, dim=-1)  # Shape: (seq_len, dim)
    return (cos, sin)


def transformer(in_tokens: torch.Tensor, weights: TransformerWeights):
    x = weights.token_emb[in_tokens]
    b, t, d = x.shape
    q_intermediate = weights.layers[0].q_proj.shape[1]
    freqs_cis = precompute_freqs_cis(q_intermediate // NUM_Q_HEADS, t)  # (cos, sin)
    for i, layer in enumerate(weights.layers):
        residual = x
        hidden = norm(x, layer.input_norm)
        hidden = attn(hidden, layer, freqs_cis, sliding_window_size=SLIDING_WINDOW_SIZE if i % 6 != 0 else None)  # Follows https://research.character.ai/optimizing-inference/
        hidden = residual + hidden

        residual = hidden
        hidden = norm(hidden, layer.post_attn_norm)
        hidden = ffn(hidden, layer)
        hidden = residual + hidden
        x = hidden

    x = norm(x, weights.final_norm)
    x = x @ weights.lm_head
    return x


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # Download the official repo weights
    state_dict = torch.load("Meta-Llama-3-8B/consolidated.00.pth", map_location="cuda")
    layers = []
    n_layers = 32
    for i in range(n_layers):
        layer = LayerWeights(
            input_norm=state_dict[f"layers.{i}.attention_norm.weight"],
            post_attn_norm=state_dict[f"layers.{i}.ffn_norm.weight"],
            q_proj=state_dict[f"layers.{i}.attention.wq.weight"].t(),
            k_proj=state_dict[f"layers.{i}.attention.wk.weight"].t(),
            v_proj=state_dict[f"layers.{i}.attention.wv.weight"].t(),
            o_proj=state_dict[f"layers.{i}.attention.wo.weight"].t(),
            gate_proj=state_dict[f"layers.{i}.feed_forward.w1.weight"].t(),
            up_proj=state_dict[f"layers.{i}.feed_forward.w3.weight"].t(),
            down_proj=state_dict[f"layers.{i}.feed_forward.w2.weight"].t(),
        )
        layers.append(layer)

    weights = TransformerWeights(
        layers=layers,
        token_emb=state_dict["tok_embeddings.weight"],
        final_norm=state_dict["norm.weight"],
        lm_head=state_dict["output.weight"].t(),
    )

    prompt = "the answer to the ultimate question of life "
    in_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    for _ in range(10):
        out = transformer(in_tokens, weights)
        next_token = torch.argmax(out[:, -1, :])
        in_tokens = torch.cat((in_tokens, next_token.unsqueeze(0).unsqueeze(0)), dim=1)

    del weights
    del state_dict

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.bfloat16, device_map="auto", _attn_implementation="eager", use_cache=False)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=10, num_beams=1, do_sample=False)
    print("Ours:", tokenizer.decode(in_tokens[0].tolist()))
    print("Ref:", tokenizer.decode(outputs[0]))
