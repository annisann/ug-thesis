import skorch
from train_utils import train, evaluate
from Model import *
from Dataset import *
from PretrainedEmbeddings import *
from utils import *

import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
import torch.nn as nn
import torch.optim as optim

# from skorch import NeuralNet # atau classifier?
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer

n_utterances = 2
max_seq_length = 10
N_EPOCH = 5

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
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

dim = 50
pretrain = PretrainedEmbeddings()
config = {
    # ENCODER
    'pretrained_embeddings': None,
    'freeze_embeddings': True,
    'en_hidden_size': max_seq_length,

    # EMO RECOG
    'num_layers': 1,
    'hidden_size': 16,
    'lr': 0.01
    }

def train_and_evaluate(config, train_data, val_data):

    run_dir = create_run_dir()

    # for train
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_Attention(config) #.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} trainable parameters.')

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # best_val_loss = float('inf')
    # RUN TIMEEEEEEEE
    train_loss, val_loss = [], []
    for epoch in range(N_EPOCH):
        start_time = time.time()

        epoch_train_loss = train(model, train_data, criterion, optimizer)#, device)
        epoch_val_loss = evaluate(model, val_data, criterion)#, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'model2.pt')

        print(f'[INFO] Epoch {epoch + 1}/{N_EPOCH} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train loss: {epoch_train_loss}')
        print(f'Val loss: {epoch_val_loss}')
        print('-'*100)

    save_plots(train_loss, val_loss,
               f'{run_dir}/loss.png')


def run_search(data):

    search_folder = create_search_dir()
    model = BiLSTM_Attention(config)
    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])


    net = NeuralNetRegressor(module=BiLSTM_Attention,
                    max_epochs=N_EPOCH,
                    optimizer=optimizer,
                    criterion=criterion,
                    lr=config['lr'])

    params = {
        'lr': [0.0001, 0.001, 0.01, 0.1],
        'module__num_layers': [1, 2],
        'module__hidden_size': [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10],
        'module__embeddings_dim': [50, 300],
        'max_epochs': list(range(10, 50, 10))
    }

    mean_squared_error = make_scorer(mean_squared_error)
    grid_search = GridSearchCV(net, params, cv=1, scoring=mean_squared_error)

    COUNTER=0
    SEARCH_BATCH=2

    for input in data:
        COUNTER += 1
        X, y = input[0], (input[1], input[2], input[3])
        outputs = grid_search.fit(X, y)

        if COUNTER == SEARCH_BATCH:
            break


    print(f"best score: {grid_search.best_score_}, best params: {grid_search.best_params_}")
    save_best_hyperparam(grid_search.best_score_, f"{search_folder}/best_params.yml")
    save_best_hyperparam(grid_search.best_params_, f"{search_folder}/best_params.yml")


if __name__ == '__main__':
    config = {
        # ENCODER
        'pretrained_embeddings': None,
        'freeze_embeddings': True,
        'en_hidden_size': max_seq_length,

        # EMO RECOG
        'num_layers': 1,
        'hidden_size': 16,
        'lr': 0.01
    }
    vocab, config['pretrained_embeddings'] = pretrain.load_from_file(dim)

    dataset = ConvEmoRecogDataset(utterance_num=n_utterances,
                                  vocab=vocab,
                                  max_seq_length=max_seq_length)
    traindf, valdf, testdf = dataset.load_dataset()

    train_data = prepare_data(traindf, n_utterances)
    val_data = prepare_data(valdf, n_utterances)
    # test_data = prepare_data(test, n_utterances)

    print('Train and evaluate')
    train_and_evaluate(config, train_data, val_data)
    print('Search')
    run_search(val_data)
