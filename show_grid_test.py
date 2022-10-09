import numpy as np
import matplotlib.pyplot as plt

nrows = 26
ncols = 26

Cellid = [2, 4 ,5, 11 ,45 ,48 ,98]
Cellval = [20, 45 ,55, 77,45 ,30 ,15]

data = np.zeros(nrows*ncols)
data[Cellid] = Cellval

data = np.ma.array(data.reshape((nrows, ncols)), mask=data==0)

fig, ax = plt.subplots()
ax.imshow(data, cmap="Greens", origin="lower", vmin=0)

# optionally add grid
ax.set_xticks(np.arange(ncols+1)-0.5, minor=True)
ax.set_yticks(np.arange(nrows+1)-0.5, minor=True)
ax.grid(which="minor")
ax.tick_params(which="minor", size=0)

plt.show()

def plot_attention(weights, head_number):
    # weights: (batch_size, num_heads, seq_len, seq_len)
    seq_len = weights.shape[2]
    grid_data = weights.detach().numpy()[0, head_number, :, :]
    fig, ax = plt.subplots()
    ax.imshow(grid_data, cmap="Greens", origin="lower", vmin=0)

    # Add grid

    ax.set_xticks(np.arange(ncols+1)-0.5, minor=True)
    ax.set_yticks(np.arange(nrows+1)-0.5, minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", size=0)

    plt.show()


