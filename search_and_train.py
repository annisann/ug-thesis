# import skorch
from train_utils import train, evaluate
from Model import *
from Dataset import *
from PretrainedEmbeddings import *
from utils import *

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder

from skorch import NeuralNet # atau classifier?
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

pretrain = PretrainedEmbeddings()
# config = {
#     'n_utterances': 2,
#     'max_seq_length': 10,
#     'dim': 200,
#     'n_epoch': 20,
#     'with_attention': False,
#
#     # ENCODER
#     'pretrained_embeddings': None,
#     'freeze_embeddings': True,
#     'en_hidden_size': 16,
#
#     # EMO RECOG
#     'num_layers': 1,
#     'hidden_size': 16,
#     'lr': 0.01
#     }

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', dest='n_epochs', default=20, type=int,
                    help='number of epochs to train network')
parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.01, type=float,
                    help="optimizer's learning rate")
parser.add_argument('-utt', '--n-utterances', dest='n_utterances', default=2, type=int,
                    help='number of utterances')
parser.add_argument('-dim', '--embed-dim', dest='dim', default=50, type=int,
                    help='embedding dimension size')
parser.add_argument('-l', '--n-layers', dest='num_layers', default=1, type=int,
                    help='number of bilstm layers')
parser.add_argument('-s', '--seq-len', dest='max_seq_len', default=10, type=int,
                    help='sequence length for encoder')
parser.add_argument('-eh', '--en-hidden', dest='en_hidden_size', default=16, type=int,
                    help='encoder hidden size')
parser.add_argument('-hs', '--hidden', dest='hidden_size', default=16, type=int,
                    help='hidden size')
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=16, type=int,
                    help='batch size of the data')
parser.add_argument('--attention', dest='with_attention', action='store_true')
parser.add_argument('--early-stop', dest='early_stopping', action='store_true')
config = vars(parser.parse_args())

def count_time(elapsed_time):
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def prepare_data(dataset, n_utterances):
    """
    :param dataset: type of dataset
    :param n_utterances: num of utterances for input
    :return: tuple of [seq_token_i, ..., seq_token_n], mean(V), mean(A), mean(D)
    """
    seq = []

    for i in range(len(dataset)):
        if i % n_utterances == 0:
            data = tuple(
                (np.array(list(dataset.seq[i:i + n_utterances])),
                 torch.from_numpy(np.array(sum(list(dataset.v[i:i + n_utterances])) / n_utterances, dtype=np.float32)),
                 torch.from_numpy(np.array(sum(list(dataset.a[i:i + n_utterances])) / n_utterances, dtype=np.float32)),
                 torch.from_numpy(np.array(sum(list(dataset.d[i:i + n_utterances])) / n_utterances, dtype=np.float32))
                 )
            )
            seq.append(data)
    return seq


def train_and_evaluate(config, train_data, val_data):

    run_dir = create_run_dir()

    save_hyperparameters(config, f'{run_dir}/hyperparameters.yml')

    # save_config(config,
    #             f'{run_dir}/config.txt')

    # for train
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_Attention(config, embeddings) #.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} trainable parameters.')


    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # lr_finder = LRFinder(model, optimizer, criterion)
    # lr_finder.range_test(train_data, val_loader=val_data, end_lr=1, num_iter=50, step_mode="linear")
    # lr_finder.plot(log_lr=False)
    # lr_finder.reset()

    # best_val_loss = float('inf')
    train_loss, val_loss = [], []
    train_time = []
    for epoch in range(config['n_epochs']):
        start_time = time.time()

        epoch_train_loss = train(model, train_data, criterion, optimizer)#, device)
        epoch_val_loss = evaluate(model, val_data, criterion)#, device)

        end_time = time.time()

        epoch_mins, epoch_secs = count_time(end_time-start_time)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_time.append(end_time-start_time)

        if config['early_stopping']:
            early_stopping = EarlyStop()
            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                break

        print('-' * 100)
        print(f'[INFO] Epoch {epoch + 1}/{config["n_epochs"]} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train loss: {epoch_train_loss}')
        print(f'Val loss: {epoch_val_loss}')

    # Total training time
    total_train_time = sum(train_time)
    train_mins, train_secs = count_time(total_train_time)
    print(f'[INFO] Total Training Time: {train_mins}m {train_secs}s')

    # Save file
    save_plots(train_loss, val_loss,
               f'{run_dir}/loss.png')

    torch.save(model.state_dict(),
               f'{run_dir}/state_dict.pt')


if __name__ == '__main__':

    vocab, embeddings = pretrain.load_from_file(config['dim'])

    dataset = ConvEmoRecogDataset(utterance_num=config['n_utterances'],
                                  vocab=vocab,
                                  max_seq_length=config['max_seq_len'])
    traindf, valdf, testdf = dataset.load_dataset()

    train_seq = prepare_data(traindf, config['n_utterances'])
    val_seq = prepare_data(valdf, config['n_utterances'])
    test_seq = prepare_data(testdf, config['n_utterances'])

    train_data = NUtterancesDataset(train_seq)
    val_data = NUtterancesDataset(val_seq)
    test_data = NUtterancesDataset(test_seq)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=True)

    print('Train and evaluate')
    train_and_evaluate(config, train_seq, val_seq)
    # train_and_evaluate(config, train_loader, val_loader)
    # print('Search')
    # run_search(val_data)
