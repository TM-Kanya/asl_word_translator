import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import re

def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    
    pattern = r"bs(\d+)_lr([0-9.]+)_epoch(\d+)"
    match = re.search(pattern, path)

    if match:
        batch_size = int(match.group(1))
        learning_rate = float(match.group(2))
        epoch = int(match.group(3))
        
    train_acc = np.loadtxt("{}.pth_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}.pth_val_acc.csv".format(path))
    
    plt.figure(figsize=(8, 6))  # Adjust figure size
    plt.title(f"Train vs Validation Accuracy\nBatch Size: {batch_size}, LR: {learning_rate}, Epochs: {epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    n = len(train_acc)  # number of epochs
    epochs = range(1, n + 1)

    plt.plot(epochs, train_acc, label="Train", linestyle="-")
    plt.plot(epochs, val_acc, label="Validation", linestyle="-")

    plt.xticks(np.linspace(1, n, min(n, 10), dtype=int))  # More detailed x-axis
    plt.yticks(np.linspace(min(min(train_acc), min(val_acc)), max(max(train_acc), max(val_acc)), 10))  # More detailed y-axis

    plt.grid(True, linestyle="--", alpha=0.6)  # Add grid with dashed lines
    plt.legend()
    plt.show()

def name(name, batch_size, learning_rate, epoch):
  return "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)

# model1_path = name("cnn", 32, 0.01, 14)
# plot_training_curve(model1_path)

# path2 = name("cnn", 256, 0.005, 24)
# plot_training_curve(path2)

# path3 = name("cnn", 16, 0.01, 49)
# plot_training_curve(path3)

# path4 = name("cnn", 32, 0.01, 14)
# plot_training_curve(path4)

# path = name("cnn", 16, 0.05, 29)
# plot_training_curve(path)

# Note: train with a higher learnign rate ~0.01.... and more epochs


# path = name("cnn", 16, 0.05, 119)
# plot_training_curve(path)


# path = name("cnn", 64, 0.015, 69)
# plot_training_curve(path)


# path = name("cnn", 16, 0.025, 119)
# plot_training_curve(path)

# path = name("cnn", 64, 0.01, 79)
# plot_training_curve(path)


path = name("cnn", 65, 0.01, 149)
plot_training_curve(path)




