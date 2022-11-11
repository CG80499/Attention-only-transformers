import random

import matplotlib.pyplot as plt
import torch

from toy_transformers import LetterDataset, TrainingLog, Transformer, mean, alphabet, plot_data

# Seeds

SEED = 314159

torch.manual_seed(SEED)
random.seed(SEED)

model = Transformer(
    n_vocab=26, 
    d_model=32,
    n_heads=4,
    seq_len=24,
    n_layers=2,
    use_layer_norm=False,
    use_smear=False,
)

folder = "checkpoints/two_layer_transformer"

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

training_log = TrainingLog(
    hyperparameters= model.hyperparameters | {"lr": 1e-3, "weight_decay": 1.0},
    seed=SEED, 
    out_file=f"{folder}/training_log.json",
)

train_dataset = LetterDataset(6, 24)
test_dataset = LetterDataset(6, 24)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

plt.title('Test vs Train Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

total_steps = 10000

train_loss = [3.3]
test_loss = []

for step in range(total_steps):

    print(f"Step {step+1}/{total_steps} Loss: {mean(train_loss[-100:])}")
    model.train()
    one_hot_input_train = next(iter(train_loader))
    pred_logits = model(one_hot_input_train)
    # pred_logits has shape (batch_size, seq_len, n_vocab)
    n_tokens = len(alphabet)

    loss = loss_fn(pred_logits[:, :-1, :].reshape(-1, n_tokens), one_hot_input_train[:, 1:, :].reshape(-1, n_tokens)) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
    train_loss.append(loss.item())

    model.eval()
    one_hot_input_test = next(iter(test_loader))
    with torch.no_grad():
        pred_logits = model(one_hot_input_test)
    loss = loss_fn(pred_logits[:, :-1, :].reshape(-1, n_tokens), one_hot_input_test[:, 1:, :].reshape(-1, n_tokens))
    per_token_loss = torch.nn.CrossEntropyLoss(reduction="none")(pred_logits[:, :-1, :].reshape(-1, n_tokens), one_hot_input_test[:, 1:, :].reshape(-1, n_tokens)).detach().reshape(-1, 23)

    ICL_loss = (per_token_loss[:, -1]-per_token_loss[:, 0]).mean()
    test_loss.append(ICL_loss.item())

    training_log.add_step({
        "step": step,
        "train_loss": train_loss[-1],
        "test_loss": test_loss[-1],
        "per_token_loss": per_token_loss.mean(dim=0).tolist(),
    })

    plot_data(train_loss, test_loss, step+1)

    if step % 256 == 0:
        training_log.save()
        torch.save(model.state_dict(), f"{folder}/model_step_{step}.pt")

plot_data(train_loss, test_loss, step+1)
plt.show()
