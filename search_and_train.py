from train_utils import train, evaluate
from Model import *
from Dataset import *
from PretrainedEmbeddings import *
from utils import *
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

pretrain = PretrainedEmbeddings()

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', dest='n_epochs', default=20, type=int,
                    help='number of epochs to train network')
parser.add_argument('-utt', '--n-utterances', dest='n_utterances', default=2, type=int,
                    help='number of utterances')
parser.add_argument('-opt', '--optimizer', dest='optimizer', default='adam', type=str,
                    help='type of optimizer')
parser.add_argument('-lr', '--learning-rate', dest='lr', default=0.025, type=float,
                    help="optimizer's learning rate")
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=64, type=int,
                    help='batch size of the data')
parser.add_argument('-dim', '--embed-dim', dest='dim', default=50, type=int,
                    help='embedding dimension size')
parser.add_argument('-s', '--seq-len', dest='max_seq_len', default=40, type=int,
                    help='sequence length for encoder')
parser.add_argument('-eh', '--en-hidden', dest='en_hidden_size', default=16, type=int,
                    help='encoder hidden size')
parser.add_argument('--en-layers', dest='en_n_layer', default=1, type=int,
                    help='number of bilstm layers on encoder')
parser.add_argument('--dropout', dest='embedding_dropout_rate', default=0.0, type=float,
                    help='dropout rate after embedding layer')
parser.add_argument('-hs', '--hidden', dest='hidden_size', default=64, type=int,
                    help='hidden size of bilstm on bilstm-attention')
parser.add_argument('-l', '--n-layers', dest='num_layers', default=1, type=int,
                    help='number of bilstm layers on bilstm-attention')

parser.add_argument('--attention', dest='with_attention', action='store_true', default=True)
parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true')
parser.add_argument('--early-stop', dest='early_stopping', action='store_true')
config = vars(parser.parse_args())

def train_and_evaluate(config, train_data, val_data):
    run_dir = create_run_dir()

    save_hyperparameters(config, f'{run_dir}/hyperparameters.yml')

    # save_config(config,
    #             f'{run_dir}/config.txt')

    # for train
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BiLSTM_Attention(config, embeddings)  # .to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {total_params:,} trainable parameters.')

    # loss function
    criterion = nn.MSELoss()

    # optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # see init weights
    with open(f"{run_dir}/weights.txt", "w") as f:
        f.write(f"========== INIT WEIGHTS ==========\n")
        for name, param in model.named_parameters():
            # print(f"{name}\n{param}")
            f.write(f"{name}\n{param}\n")

    # best_val_loss = float('inf')
    train_loss, val_loss = [], []
    train_time = []
    for epoch in range(config['n_epochs']):
        start_time = time.time()

        epoch_train_loss = train(model, train_data, criterion, optimizer)  # , device)
        epoch_val_loss = evaluate(model, val_data, criterion)  # , device)

        end_time = time.time()

        epoch_hours, epoch_mins, epoch_secs = count_time(end_time - start_time)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        train_time.append(end_time - start_time)

        # save weights
        with open(f"{run_dir}/weights.txt", "a") as f:
            f.write(f"\n========== EPOCH {epoch + 1} ==========\n")
            for name, param in model.named_parameters():
                # print(f"{name}\n{param}")
                f.write(f"{name}\n{param}\n")

        if config['lr_scheduler']:
            lr_scheduler = LRScheduler(optimizer)
            lr_scheduler(epoch_val_loss)
            current_lr = get_lr(optimizer)
            print(current_lr)
            with open(f"{run_dir}/learning_rate.txt", "a") as f:
                f.write(f"Epoch {epoch}: {current_lr}\n")

        if config['early_stopping']:
            early_stopping = EarlyStop()
            early_stopping(epoch_val_loss)
            if early_stopping.early_stop:
                break

        print('-' * 100)
        print(f'[INFO] Epoch {epoch + 1}/{config["n_epochs"]} | Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s')
        print(f'Train loss: {epoch_train_loss}')
        print(f'Val loss: {epoch_val_loss}')

    # Total training time
    total_train_time = sum(train_time)
    train_hours, train_mins, train_secs = count_time(total_train_time)
    print(f'[INFO] Total Training Time: {train_hours}h {train_mins}m {train_secs}s')

    # Save file
    save_train_time(train_time,
                    f'{run_dir}/train_time.txt')

    save_losses(train_loss, val_loss,
                f'{run_dir}/loss.txt')

    save_plots(train_loss, val_loss,
               f'{run_dir}/loss.png')

    torch.save(model.state_dict(),
               f'{run_dir}/state_dict.pt')


if __name__ == '__main__':
    vocab, embeddings = pretrain.load_from_file(config['dim'])
    print(embeddings.shape)

    dataset = ConvEmoRecogDataset(utterance_num=config['n_utterances'],
                                  vocab=vocab,
                                  max_seq_length=config['max_seq_len'])
    traindf, valdf, testdf = dataset.load_dataset()

    train_seq = prepare_data(traindf, config['n_utterances'])
    val_seq = prepare_data(valdf, config['n_utterances'])

    # train_data = NUtterancesDataset(train_seq)
    # val_data = NUtterancesDataset(val_seq)
    #
    # train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=True)

    print('Train and evaluate')
    train_and_evaluate(config, train_seq, val_seq) #????????????????????/