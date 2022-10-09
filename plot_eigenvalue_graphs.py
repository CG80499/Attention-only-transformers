from glob import glob
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

window = 21
order = 2

checkpoint_file = "checkpoints/one_layer_every_step"
checkpoints = glob(checkpoint_file + "/*.pt")
checkpoints.sort(key = lambda s: s.split("_")[2])
print(len(checkpoints))

log_file = "checkpoints/one_layer_every_step/training_log.json"

with open(log_file, "r") as f:
    training_log = json.load(f)

def copy_metric(values):
    return np.sum(values) / np.sum(np.abs(values))

def get_copy_metric(state_dict):
    num_heads = 4
    d_k = 32//num_heads

    W_E = state_dict["layers.0.embedding_matrix.weight"].T
    W_O = state_dict["layers.1.W_o.weight"].T
    W_V = state_dict["layers.1.W_v.weight"].T
    W_U = state_dict["head.weight"].T

    for i in range(4):
        v_head1 = W_V[:, i*d_k:(i+1)*d_k]

        o_head1 = W_O[i*d_k:(i+1)*d_k, :]


        OV_circuit1 = W_E @ v_head1 @ o_head1 @ W_U
        OV_circuit1 = OV_circuit1.detach().numpy()


        eigenvalues, _ = np.linalg.eig(OV_circuit1)

        yield copy_metric(eigenvalues).real

    direct_path = W_E @ W_U

    eigenvalues, _ = np.linalg.eig(direct_path)

    yield copy_metric(eigenvalues).real

head_eigenvalues = []

steps = training_log["step_stats"]

step_numbers = list(range(1000))

for checkpoint in tqdm(checkpoints):
    state_dict = torch.load(checkpoint)
    head_eigenvalues.append(list(get_copy_metric(state_dict)))


for head in range(4):
    plt.plot(step_numbers, uniform_filter1d([h[head]for h in head_eigenvalues], 1), label = f"Head {head+1}")

#print(step_numbers, [h[0] for h in head_eigenvalues])
plt.plot(step_numbers, uniform_filter1d([h[4]  for h in head_eigenvalues], 1), label = "Direct path")

#plt.plot([s["step"] for s in steps], [s["train_loss"] for s in steps], label = "Loss")

plt.show()