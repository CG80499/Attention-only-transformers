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
    use_smear=False,
)

model.eval()

pattern = "ABCDEF"*2

checkpoint_file = "checkpoints/one_layer_transformer/model_step_9984.pt"

state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

#del state_dict["layers.1.norm.weight"], state_dict["layers.1.norm.bias"] # Old key names

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
        print("Next token prob", logits[0][index])
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
#Embedding circuit copy metric: (-0.9995849-1.2085145e-10j)
#Head 1 Copy metric:  (0.6860927-2.1031352e-16j)
#Head 2 Copy metric:  (0.85695416+2.0477617e-10j)
#Head 3 Copy metric:  (0.98481447+1.7858229e-11j)
#Head 4 Copy metric:  (0.5779195+4.070983e-11j)

# Cross entropy loss: 2.6064820140600204 Loss with all heads enabled
# Cross entropy loss: 2.805475741624832 Loss with head 1 disabled
# Cross entropy loss: 2.835808351635933 Loss with head 2 disabled
# Cross entropy loss: 2.8343328833580017 Loss with head 3 disabled
# Cross entropy loss: 2.8077731877565384

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