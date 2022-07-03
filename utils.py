import torch
import matplotlib
import matplotlib.pyplot as plt
import glob
import os


def count_time(time):
    sec = int(time % (24 * 3600))
    hour = int(sec / 3600)
    sec %= 3600
    min = int(sec / 60)
    sec %= 60
    return hour, min, sec


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
    plt.xticks(range(1, len(valid_loss)))
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)


def save_hyperparameters(text, path):
    with open(path, 'w') as file:
        keys = list(text.keys())
        for key in keys:
            file.writelines(f"{key}: {text[key]}\n")


def save_losses(train_loss, val_loss, path):
    with open(path, 'w') as file:
        file.write("========== Train Loss ==========\n")
        file.writelines('\n'.join(str(loss) for loss in train_loss))
        file.write("\n========== Validation Loss ==========\n")
        file.writelines('\n'.join(str(loss) for loss in val_loss))


def save_train_time(time_per_epoch, path):
    """
    Save training time in txt file.
    :param time_per_epoch:List -> elapsed time per epoch
    :param path:Str -> file path
    :return:
    """
    total_train_time = sum(time_per_epoch)
    train_hours, train_mins, train_secs = count_time(total_train_time)

    decode_time_per_epoch = []
    for t in time_per_epoch:
        hour, min, sec = count_time(t)
        decode_time_per_epoch.append(f"{hour}h {min}m {sec}s")

    with open(path, 'w') as file:
        file.writelines("\n".join(decode_time_per_epoch))
        file.write(f"\nTotal training time: {train_hours}h {train_mins}m {train_secs}s")


def create_run_dir():
    n_run_dirs = len(glob.glob('outputs/run_*'))
    run_dir = f"outputs/run_{n_run_dirs + 1}"
    os.makedirs(run_dir)
    return run_dir


def create_test_dir():
    n_test_dirs = len(glob.glob('tests/test_*'))
    test_dir = f"tests/test_{n_test_dirs + 1}"
    os.makedirs(test_dir)
    return test_dir


def create_search_dir():
    n_search_dirs = len(glob.glob('outputs/search_*'))
    search_dir = f"outputs/search_{n_search_dirs + 1}"
    os.makedirs(search_dir)
    return search_dir


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
        if self.best_loss == None:
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


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=2, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
