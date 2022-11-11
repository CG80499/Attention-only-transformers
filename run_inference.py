import torch
from toy_transformers import Transformer
from toy_transformers import create_one_hot, alphabet2index, LetterDataset
import numpy as np

model = Transformer(
    n_vocab=26, 
    d_model=32,
    n_heads=4,
    seq_len=24,
    n_layers=2,
    use_layer_norm=False,
    use_smear=False,
)

# Total: 26*32+4*32**2+4*32**2+26*32 = 9856

model.eval()

pattern = "AXAAAA"*2

checkpoint_file = "checkpoints/two_layer_transformer/model_step_9984.pt"

state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

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
        print("Next token prob", logits[0][index][next_char].item())
        if next_char == 26:
            break
        string += index2alphabet[next_char]
        input = create_one_hot(pad_string(string, length=max_length)).unsqueeze(0)
        if len(string) == max_length:
            break
        with torch.no_grad():
            logits = model(input).softmax(dim=-1)
    return string

print(greedy_decode(pattern, 24))
print((pattern*(24//6+1))[:24])
"""
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
test_dataset = LetterDataset(6, 24)
losses = []
for i in range(1):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    one_hot_input_test = next(iter(test_loader))

    with torch.no_grad():
        pred_logits = model(one_hot_input_test)
    loss = loss_fn(pred_logits[:, :-1, :].reshape(-1, 26), one_hot_input_test[:, 1:, :].reshape(-1, 26))

    losses.append(loss.item())

print("Cross entropy loss:", sum(losses)/len(losses))

"""
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
# All heads added together:  Copy metric:  (0.9997909-4.1352373e-11j)

# Cross entropy loss: 2.6064820140600204 Loss with all heads enabled
# Cross entropy loss: 2.805475741624832 Loss with head 1 disabled
# Cross entropy loss: 2.835808351635933 Loss with head 2 disabled
# Cross entropy loss: 2.8343328833580017 Loss with head 3 disabled
# Cross entropy loss: 2.8077731877565384 Loss with head 4 disabled


# Smeared key

# Embedding circuit copy metric: (-0.9998577+0j)   
# Head 1 Copy metric:  (-0.71228284+2.4907904e-16j)
# Head 2 Copy metric:  (0.99999994+0j)
# Head 3 Copy metric:  (1.0000001+1.0977872e-16j)  
# Head 4 Copy metric:  (-0.856813-7.7495547e-16j)  
"""

state_dict = model.state_dict()

num_heads = 4
d_k = 32//num_heads

W_E = state_dict["layers.0.embedding_matrix.weight"].T
W_O = state_dict["layers.1.W_o.weight"].T
W_V = state_dict["layers.1.W_v.weight"].T
W_U = state_dict["head.weight"].T

W_Q = state_dict["layers.2.W_q.weight"].T
W_K = state_dict["layers.2.W_k.weight"].T

def copy_metric(values):
    return np.sum(values) / np.sum(np.abs(values))

def average_magnitude(values):
    return np.mean(np.abs(values))

embedding_circuit = W_E @ W_U
embedding_circuit = embedding_circuit.detach().numpy()


eigenvalues, eigenvectors = np.linalg.eig(embedding_circuit)
#print("Embedding circuit copy metric:", copy_metric(eigenvalues))

circuits = []
for i in range(4):

    Q_head = W_Q[:, i*d_k:(i+1)*d_k]

    K_head = W_K[:, i*d_k:(i+1)*d_k]

    i = 3
    v_head1 = W_V[:, i*d_k:(i+1)*d_k]

    o_head1 = W_O[i*d_k:(i+1)*d_k, :]

    QK_circuit1 = K_head @ Q_head.T

    #OV_circuit1 = W_E @ v_head1 @ o_head1 @ W_U

    OV_circuit1 = W_E @ v_head1 @ o_head1 @ QK_circuit1 @ W_E.T
    OV_circuit1 = OV_circuit1.detach().numpy()
    circuits.append(OV_circuit1)


    eigenvalues, eigenvectors = np.linalg.eig(OV_circuit1)

    print(f"Head {i+1} Copy metric: ", copy_metric(eigenvalues))
    #print(f"Head {i+1} Average magnitude: ", average_magnitude(eigenvalues))

    eigenvectors = eigenvectors.real
    eigenvectors = [v/np.linalg.norm(v) for v in eigenvectors]


"""
    #print(f"Head {i+1} Eigenvectors:", eigenvectors)

    #print("Eigenvalues: ", eigenvectors)
#print([(v**2).sum() for v in eigenvectors])

#circuit = sum(circuits)

#eigenvalues, eigenvectors = np.linalg.eig(circuit)

#print("Copy metric: ", copy_metric(eigenvalues))

# 2-layer transformer (layer 1)
# Embedding circuit copy metric: (-0.9836473+1.410438e-09j)
# Layer 1:
# Head 1 Copy metric:  (-0.9999998-5.327597e-16j)
# Head 2 Copy metric:  (-0.94031566-2.6753054e-16j)
# Head 3 Copy metric:  (-0.9329939+0j)
# Head 4 Copy metric:  (-0.9999997+7.125459e-16j)
# Layer 2:
# Head 1 Copy metric:  (0.9998253+1.1311128e-16j)
# Head 2 Copy metric:  (0.9999484+0j)
# Head 3 Copy metric:  (0.99989885+4.20817e-13j)
# Head 4 Copy metric:  (0.99702746+1.5782087e-10j)

# Baseline Cross entropy loss: 1.3844698667526245

# Cross entropy loss with head 1 in layer 1 disabled: 2.2648746967315674
# Cross entropy loss with head 2 in layer 1 disabled: 2.457996368408203
# Cross entropy loss with head 3 in layer 1 disabled: 2.7364978790283203
# Cross entropy loss with head 4 in layer 1 disabled: 2.4093422889709473

# 2nd layer just using direct path for attention: 3.1235387325286865
# 2nd layer just using direct path for attention + K composition: 2.518188714981079
# 2nd layer just using direct path for attention + Q composition: 2.6181252002716064
# 2nd layer just using direct path for attention + K composition + Q compositions: 1.3905218839645386

# Matching in from head 1!
# Head 1 Copy metric:  (0.89388597-2.2985782e-09j)
# Head 1 Copy metric:  (-0.6828638+0j)
# Head 1 Copy metric:  (-0.92538565+3.692424e-11j)
# Head 1 Copy metric:  (0.70200497+6.051526e-16j)

# Matching in from head 2!
# Head 2 Copy metric:  (0.99992883+6.803108e-15j)
# Head 2 Copy metric:  (0.999999-4.2807158e-14j)
# Head 2 Copy metric:  (0.99996066+0j)
# Head 2 Copy metric:  (0.9999983-3.4481166e-13j)

# Matching in from head 3!
# Head 3 Copy metric:  (-0.3635076+0j)
# Head 3 Copy metric:  (-0.63910097+0j)
# Head 3 Copy metric:  (0.9450826-2.4281412e-09j)
# Head 3 Copy metric:  (-0.4290508+5.9449112e-09j)

# Matching in from head 4!
# Head 4 Copy metric:  (-0.9442232+0j)
# Head 4 Copy metric:  (0.91947246-5.316366e-09j)
# Head 4 Copy metric:  (0.2616711-2.5381839e-08j)
# Head 4 Copy metric:  (0.32606995+0j)