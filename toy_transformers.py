import json
import matplotlib.pyplot as plt
from positional_embeddings import sinusoidal_embeddings
import torch
import string
import random

alphabet = list(string.ascii_lowercase.upper())
alphabet2index = {letter: index for index, letter in enumerate(alphabet)}

LARGE_NUMBER = 1e9

def create_sample(length, pattern_len=6):
    pattern = "".join([random.choice(alphabet) for _ in range(pattern_len)])
    return (pattern*(length//pattern_len+1))[:length] 

def create_fixed_sample(length, pattern="ABCDEF"):
    pattern_len = len(pattern)
    return (pattern*(length//pattern_len+1))[:length] 

def tokenize(text):
    return [alphabet2index[c] for c in text] 

def create_one_hot(text):
    return torch.eye(len(alphabet))[tokenize(text)] # Torch.eye creates an n x n identity matrix.
    
def batch_generator(l, batch_size):
    for i in range(0, len(l), batch_size):
        yield l[i:i+batch_size]

def mean(l):
    return sum(l)/len(l) if len(l) > 0 else 0

def plot_attention(weights, head_number):
    import numpy as np
    # weights: (batch_size, num_heads, seq_len, seq_len)
    seq_len = weights.shape[2]
    grid_data = weights.detach().mean(dim=0).numpy()[head_number, :, :]
    #grid_data = weights.detach().numpy()[0, head_number, :, :]
    # Normalize grid_data
    grid_data = grid_data / grid_data.max(axis=0, keepdims=True)
    #grid_data = np.clip(grid_data, 0, 1/3)
    fig, ax = plt.subplots()
    ax.imshow(grid_data, cmap="Blues", origin="upper", vmin=0) 

    # Add grid

    ax.set_xticks(np.arange(seq_len+1)-0.5, minor=True)
    ax.set_yticks(np.arange(seq_len+1)-0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    plt.show()
    #head_number += 1
    #plt.savefig(f'images/two_layer_head_{head_number}_layer_2.png', bbox_inches='tight')

def attention(K, Q, V, mask):
    # K, Q have shape (batch_size, n_heads, seq_len, d_k)
    # V has shape (batch_size, n_heads, seq_len, d_v)
    # Note d_k = d_v
    seq_len, d_k = K.shape[-2:]
    positional_embeddings = sinusoidal_embeddings(seq_len, d_k).to(K.device)
    K, Q = K + positional_embeddings, Q + positional_embeddings # Added after multiplication by W_k, W_q to avoid putting positional embeddings in the residual stream. 
    scores = torch.matmul(Q, K.transpose(2, 3)) / d_k**0.5 + mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V)

class Embedding(torch.nn.Module):

    def __init__(self, n_vocab, d_model):
        super().__init__()
        self.embedding_matrix = torch.nn.Linear(n_vocab, d_model, bias=False)

    def forward(self, x):
        # x has shape (batch_size, seq_len, n_vocab)
        # embedding_matrix has shape (n_vocab, d_model)
        # output has shape (batch_size, seq_len, d_model)
        return self.embedding_matrix(x)

class FudgedLayerNorm(torch.nn.Module):

    def __init__(self, seq_len, d_model, eps=1e-6, alpha=0.99):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_model))
        self.alpha = alpha
        self.var = torch.ones(1, seq_len, 1)
        self.register_buffer("fudged_layernorm_var", self.var)

    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        # output has shape (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        if self.training:
            avg_batch_var = x.var(dim=-1, keepdim=True).mean(dim=0, keepdim=True).detach()
            self.var = self.alpha*self.var + (1-self.alpha)*avg_batch_var
        return self.gamma * (x - mean) / (self.var + self.eps)**0.5 + self.beta

class Smear(torch.nn.Module):

    def __init__(self, n_heads, seq_len):
        super().__init__()
        self.alpha_values = torch.nn.Parameter(torch.ones(1, n_heads, seq_len-1, 1)) # This corresponds an inital weighting of 73% to the first key and 27% to the second key.
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, K):
        # K has shape (batch_size, n_heads, seq_len, d_k)
        smeared_K = K[:, :, 1:, :]*(self.sigmoid(self.alpha_values)) + K[:, :, :-1, :]*(1-self.sigmoid(self.alpha_values))
        return torch.cat([K[:, :, 0:1, :], smeared_K], dim=2)
        

class DecoderLayer(torch.nn.Module):
    
    def __init__(self, d_model, n_heads, seq_len, use_layer_norm=False, use_smear=False, layer_number=-1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.dv = d_model // n_heads
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_model = d_model
        self.mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)*(-LARGE_NUMBER)
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False) # Note: d_model = d_k * n_heads
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False) # It's easier to do the split in the forward pass.
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.norm = torch.nn.LayerNorm(d_model) if use_layer_norm else lambda x: x
        self.smear = Smear(n_heads, seq_len) if use_smear else lambda x: x
        self.layer_number = layer_number

    
    def forward(self, X, direct_path):
        # X has shape (batch_size, seq_len, d_model)
        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        # Q, K, V have shape (batch_size, seq_len, d_model)
        if self.layer_number == 1:
            mean_Q, mean_K = Q[:256].mean(dim=0).detach(), K[:256].mean(dim=0).detach()
            # Fix K
            #K[256:512, :, :] = mean_K
            # Fix Q
            Q[256:512, :, :] = mean_Q
            #Q = self.W_q(direct_path)
            #K = self.W_k(direct_path)
        Q, K, V = Q.view(-1, self.seq_len, self.n_heads, self.d_k), K.view(-1, self.seq_len, self.n_heads, self.d_k), V.view(-1, self.seq_len, self.n_heads, self.dv)
        # Q, K, V have shape (batch_size, seq_len, n_heads, d_k)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        # Q and K have shape (batch_size, n_heads, seq_len, d_k) and V has shape (batch_size, n_heads, seq_len, d_v)
        # Apply smeared key if used.
        K = self.smear(K)
        multi_head_attention = attention(K, Q, V, self.mask)
        # multi_head_attention has shape (batch_size, n_heads, seq_len, d_v)
        multi_head_attention = multi_head_attention.transpose(1, 2).contiguous()
        # multi_head_attention has shape (batch_size, seq_len, n_heads, d_v)
        multi_head_attention = multi_head_attention.view(-1, self.seq_len, self.d_model)
        # multi_head_attention has shape (batch_size, seq_len, d_model)
        Y = self.W_o(multi_head_attention)+X # Don't forget the residual connection!
        # Y has shape (batch_size, seq_len, d_model)
        Y = self.norm(Y)
        return Y

class Transformer(torch.nn.Module):

    def __init__(self, n_vocab, d_model, n_heads, seq_len, n_layers, use_layer_norm=False, use_smear=False):
        super().__init__()
        self.embedding = Embedding(n_vocab, d_model)
        self.head = torch.nn.Linear(d_model, n_vocab, bias=False)
        self.layers = torch.nn.Sequential(
            self.embedding,
            *[DecoderLayer(d_model, n_heads, seq_len, use_layer_norm, use_smear, i) for i in range(n_layers)],
            self.head,
        )
        self.hyperparameters = {
            "n_vocab": n_vocab,
            "d_model": d_model,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "n_layers": n_layers,
            "use_layer_norm": use_layer_norm,
            "use_smear": use_smear,
        }
    
    def forward(self, x):
        direct_path = self.embedding(x).clone()
        #return self.layers(x, direct_path)
        x = self.embedding(x)
        for layer in self.layers[1:-1]:
            x = layer(x, direct_path)
        return self.head(x)

    def __str__(self):
        return "Transformer with hyperparameters: \n" + "\n".join([f"{k}: {v}" for k, v in self.hyperparameters.items()])

class TrainingLog:

    def __init__(self, hyperparameters, seed, out_file):
        self.hyperparameters = hyperparameters
        self.step_stats = []   
        self.out_file = out_file 
        self.seed = seed

    def add_step(self, step):    
        self.step_stats.append(step)

    def save(self):
        with open(self.out_file, "w") as f:
            json.dump({
                "hyperparameters": self.hyperparameters,
                "step_stats": self.step_stats,
                "seed": self.seed,
            }, f)

class LetterDataset(torch.utils.data.Dataset):

    def __init__(self, patten_len, seq_len):
        self.seq_len = seq_len
        self.pattern_len = patten_len

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return create_one_hot(create_sample(self.seq_len, pattern_len=self.pattern_len))

class TestTextDataset(torch.utils.data.Dataset):

    def __init__(self, patten_len, seq_len):
        self.seq_len = seq_len
        self.pattern_len = patten_len

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return create_one_hot(create_fixed_sample(self.seq_len, pattern="ABCDEF"))

def plot_data(train_loss, test_loss, total_steps):
    if not len(test_loss) %  128 == 0:
        return
    #train_loss = [mean(l) for l in batch_generator(train_loss, 24)]
    _test_loss = [mean(l) for l in batch_generator(test_loss, 24)]
    #plt.plot(train_loss, label='Train', color="blue")
    plt.plot(_test_loss, label='Test', color="lightblue")
    if total_steps == 1:
        plt.legend(loc="upper left")
    plt.show(block=False)
    plt.pause(0.3)