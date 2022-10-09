import torch
from toy_transformers import Transformer
from toy_transformers import create_one_hot, alphabet2index, LetterDataset
import numpy as np

model = Transformer(
    n_vocab=26, 
    d_model=32,
    n_heads=4,
    seq_len=24,
    n_layers=1,
    use_layer_norm=False,
    use_smear=True,
)

model.eval()

pattern = "ABCDEF"*2

checkpoint_file = "checkpoints/one_layer_smeared_query/model_step_9984.pt"

state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

#del state_dict["layers.1.norm.weight"], state_dict["layers.1.norm.bias"] # Old key names
state_dict["layers.1.smear.alpha_values"] = state_dict["layers.1.smeared_keys.alpha_values"]
del state_dict["layers.1.smeared_keys.alpha_values"]

model.load_state_dict(state_dict)

model.eval()

index2alphabet = {v: k for k, v in alphabet2index.items()}

def pad_string(string, length=30):
    return string + "A"*(length-len(string))

def greedy_decode(string, max_length=30):
    input = create_one_hot(pad_string(string, length=max_length)).unsqueeze(0)
    with torch.no_grad():
        logits = model(input).softmax(dim=-1)
    for i in range(max_length):
        index = len(string) - 1
        next_char = logits[0][index].argmax(dim=-1).item()
        if next_char == 26:
            break
        string += index2alphabet[next_char]
        input = create_one_hot(pad_string(string, length=max_length)).unsqueeze(0)
        if len(string) == max_length:
            break
        with torch.no_grad():
            logits = model(input).softmax(dim=-1)
    return string

#print(greedy_decode(pattern, 24))
#print((pattern*(24//6+1))[:24])


loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
test_dataset = LetterDataset(6, 24)
losses = []
for i in range(1):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True)

    one_hot_input_test = next(iter(test_loader))

    with torch.no_grad():
        pred_logits = model(one_hot_input_test)
    loss = loss_fn(pred_logits[:, :-1, :].reshape(-1, 26), one_hot_input_test[:, 1:, :].reshape(-1, 26))
    losses.append(loss.item())

print("Cross entropy loss:", sum(losses)/len(losses))


# PQRSTWPQRSTWTQSWQWWQSWQW With head 2
# PQRSTWPQRSTWOQSWQWWQSWQS Without head 2
# PQRSTWPQRSTWPQRSTWPQRSTW

# ABCDEFABCDEFBBFZBFBBDHBC With head 2
# ABCDEFABCDEFGBFZBFBBDHDH Without head 2
# ABCDEFABCDEFABCDEFABCDEF
# Copy metric head 1:  (0.99999076-4.3177517e-11j)
# Copy metric head 2:  (-0.8484091+4.2669993e-10j)
# Copy metric head 3:  (1-1.4584884e-16j)
# Copy metric head 4:  (0.9999595+2.8697897e-10j)
# Embedding circuit copy metric: (-0.99885213+4.5348645e-11j)
# All heads Cross entropy loss: 1.6893332397937775
# Head 1 disabled Cross entropy loss: 2.1365128111839295
# Head 2 disabled Cross entropy loss: 2.1365247917175294
# Head 3 disabled Cross entropy loss: 2.136558632850647
# Head 4 disabled Cross entropy loss: 2.071822009086609
"""

state_dict = model.state_dict()

num_heads = 4
d_k = 32//num_heads

W_E = state_dict["layers.0.embedding_matrix.weight"].T
W_O = state_dict["layers.1.W_o.weight"].T
W_V = state_dict["layers.1.W_v.weight"].T
W_U = state_dict["head.weight"].T

def copy_metric(values):
    return np.sum(values) / np.sum(np.abs(values))

def average_magnitude(values):
    return np.mean(np.abs(values))

embedding_circuit = W_E @ W_U
embedding_circuit = embedding_circuit.detach().numpy()


eigenvalues, eigenvectors = np.linalg.eig(embedding_circuit)
print("Embedding circuit copy metric:", copy_metric(eigenvalues))
print("Embedding circuit average magnitude:", average_magnitude(eigenvalues))

for i in range(4):
    v_head1 = W_V[:, i*d_k:(i+1)*d_k]

    o_head1 = W_O[i*d_k:(i+1)*d_k, :]


    OV_circuit1 = W_E @ v_head1 @ o_head1 @ W_U
    OV_circuit1 = OV_circuit1.detach().numpy()


    eigenvalues, eigenvectors = np.linalg.eig(OV_circuit1)

    print(f"Head {i+1} Copy metric: ", copy_metric(eigenvalues))
    print(f"Head {i+1} Average magnitude: ", average_magnitude(eigenvalues))

    eigenvectors = eigenvectors.real
    eigenvectors = [v/np.linalg.norm(v) for v in eigenvectors]

    #print("Eigenvalues: ", eigenvectors)
#print([(v**2).sum() for v in eigenvectors])

"""
