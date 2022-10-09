import matplotlib.pyplot as plt
import random, time

fake_loss_data_train = []
fake_loss_data_test = []



def plot_data(train_loss, test_loss, total_steps):
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    if total_steps == 1:
        plt.legend(loc="upper left")
    plt.show(block=False)
    plt.pause(0.1)