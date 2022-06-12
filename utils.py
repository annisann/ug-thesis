import json

import matplotlib
import matplotlib.pyplot as plt
import glob
import os

def save_plots(train_loss, valid_loss, loss_plot_path):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)


def save_config(config, path):
    with open(path, 'w') as file:
        file.write(json.dumps(dict((k, v) for k, v in config.items() if k != 'pretrained_embeddings')))


def save_hyperparameters(text, path):
    with open(path, 'w') as f:
        keys = list(text.keys())
        for key in keys:
            f.writelines(f"{key}: {text[key]}\n")


def create_run_dir():
    n_run_dirs = len(glob.glob('outputs/run_*'))
    run_dir = f"outputs/run_{n_run_dirs+1}"
    os.makedirs(run_dir)
    return run_dir


def create_search_dir():
    n_search_dirs = len(glob.glob('outputs/search_*'))
    search_dir = f"outputs/search_{n_search_dirs+1}"
    os.makedirs(search_dir)
    return search_dir


# def save_best_hyperparameters(text, path):
#     with open(path, 'a') as f:
#         f.write(f'{str(text)}\n')

class EarlyStop:
    """
    To stop the training when the loss doesn't improve after certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: n epochs to wait before stopping.
        :param min_delta: min differences between loss(n_epoch-1) and loss(n_epoch)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"[INFO] Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(">>>>> EARLY STOPPING <<<<<")
                self.early_stop = True
