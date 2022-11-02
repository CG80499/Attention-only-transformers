import torch
import numpy as np
import functools

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )
    # Note: It does not matter the way we rotate the vector by 90 degrees.
    # This clearly satisfies the property that x * rotate_half(x) = 0. Where * is the dot product.

def theta_i(i, d_model):
    return 10000 ** (2 * i / d_model)

def sin_cos_values(seq_len, d_k, d_model):
    theta_values = torch.tensor([theta_i(i//2, d_model) for i in range(2, d_k+2)]) # They start from 1 in the paper.
    angles = torch.einsum("i,j->ij", torch.arange(seq_len), theta_values) # Outer product.
    return torch.sin(angles), torch.cos(angles)

def rotary_embeddings(Q, K, d_model):
    batch_size, n_heads, seq_len, d_k = Q.shape
    sin_values, cos_values = sin_cos_values(seq_len, d_k, d_model)
    sin_values, cos_values = sin_values.to(Q.device), cos_values.to(Q.device)
    return (
        Q*cos_values + rotate_half(Q)*sin_values,
        K*cos_values + rotate_half(K)*sin_values,
    )

def sinusoidal_embeddings_numpy(seq_len, d_model):
    position_enc = np.array([ # TODO: Use torch.arange instead.
        [pos / theta_i(i, d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(seq_len)
    ])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

@functools.cache
def sinusoidal_embeddings(seq_len, d_model):
    position_enc = torch.stack([
        torch.tensor([pos / theta_i(i, d_model) for i in range(d_model)])
        if pos != 0 else torch.zeros(d_model) for pos in range(seq_len)
    ])

    position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2]) # dim 2i+1
    return position_enc