import os
import pandas as pd
import json
import numpy as np
import ast

import torch
from Model import *
from Dataset import *

import torch.nn as nn
import torch.optim as optim
import time


torch.manual_seed(42)
# LOAD PRETRAINED EMBEDDINGS
# set dimension
dim = 50
pretrain = PretrainedEmbeddings()
vocab, embeddings = pretrain.load_from_file(dim)

# print(vocab.shape, embeddings.shape)

n_utterances = 2
max_seq_length = 512
dataset = ConvEmoRecogDataset(utterance_num=n_utterances,
                              vocab=vocab,
                              max_seq_length=max_seq_length)
train, val, test = dataset.load_dataset()  # df: id, utterance, token, v, a, d, seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {  # ENCODER
    'pretrained_embeddings': embeddings,
    'freeze_embeddings': True,
    'en_hidden_size': 256,

    # EMO RECOG
    'num_layers': 1,
    'hidden_size': 256
}

model = BiLSTM_Attention(config)
print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Model has {count_parameters(model):,} trainable parameters')


# emo = model.forward(list(val.seq[3:5]))
# # print(model(val))
# # print(emo)


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


train_data = prepare_data(train, n_utterances)
val_data = prepare_data(val, n_utterances)
test_data = prepare_data(test, n_utterances)

data_dict = {'train': train_data,
             'val': val_data,
             'test': test_data}
dataset_sizes = {'train': len(train_data),
                 'val': len(val_data),
                 'test': test_data}

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
# print(torch.all(tensor.isnan()))


def train_model(data, model, criterion, optimizer):
    num_batches = len(data)
    total_loss = 0
    model.train()

    # total_loss_v, total_loss_a, total_loss_d = 0, 0, 0
    for i, batch in enumerate(data):
        print(f'DATA {i}')
        print(data[i])
        print('-'*20)

        # unpack data
        input_utterances = batch[0]
        v_act = batch[1]
        a_act = batch[2]
        d_act = batch[3]

        # reset grad
        optimizer.zero_grad()

        v_pred, a_pred, d_pred = model(input_utterances)
        print(v_pred, a_pred, d_pred)

        loss_v = criterion(v_pred, v_act)
        loss_a = criterion(a_pred, a_act)
        loss_d = criterion(d_pred, d_act)

        # loss_v.backward(retain_graph=True)
        # loss_a.backward(retain_graph=True)
        # loss_d.backward(retain_graph=True)

        loss = loss_v + loss_a + loss_d
        print(loss)

        # backprop
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        model.float()
        # update parameters
        optimizer.step()

        # total_loss_v += loss_v.item()
        # total_loss_a += loss_a.item()
        # total_loss_d += loss_d.item()
        # print(f'Loss_V:{loss_v}, Loss_A:{loss_a}, Loss_D:{loss_d}')
        # print(total_loss_v, total_loss_a, total_loss_d)
        print()

        total_loss += loss.item()
    avg_loss = total_loss/num_batches
    # avg_loss_v = total_loss_v/num_batches
    # avg_loss_a = total_loss_a/num_batches
    # avg_loss_d = total_loss_d/num_batches
    # print(f'AVG Loss V: {avg_loss_v}\nAVG Loss A: {avg_loss_a}\nAVG Loss D: {avg_loss_d}')
    print(f'Train Loss: {avg_loss}')

for i in range(10):
    print(f'Epoch {i+1}')
    print('-'*20)
    train_model(train_data[:100], model, criterion, optimizer)
    print()


# def train(model, criterion, optimizer, num_epochs=10):  # , optimizer, loss_function):
#     since = time.time()
#     model.train()
#     epoch_loss = 0
#
#     train_loss, val_loss = [], []
#
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch + 1}/{num_epochs}')
#         print('-' * 10)
#
#         for phase in ['train', 'val']:
#             running_loss = 0.0
#             if phase == 'train':
#                 model.train()
#
#                 for batch in data_dict[phase]:
#                     optimizer.zero_grad()
#
#                     with torch.set_grad_enabled(phase == 'train'):
#                         input_utterances = batch[0]
#                         v = torch.autograd.Variable(batch[1])
#                         a = torch.autograd.Variable(batch[2])
#                         d = torch.autograd.Variable(batch[3])
#
#                         v_pred, a_pred, d_pred = model(input_utterances)
#
#                         loss_v = criterion(v_pred, v)
#                         loss_a = criterion(a_pred, a)
#                         loss_d = criterion(d_pred, d)
#                         # print(f'loss_V:{loss_v}')
#                         #
#                         loss = loss_v + loss_a + loss_d
#                         print(loss_v.dtype, loss_a.dtype, loss_d.dtype, loss.dtype)
#                         # print(f'loss:{loss}')
#                         loss.backward()
#                         optimizer.step()
#                     running_loss += loss.item()
#                     print(running_loss)
#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 train_loss.append(epoch_loss)
#     print(epoch_loss)

    # elif phase == 'val':
    #     model.eval()
    #     optimizer.zero_grad()
    # elapsed_time = time.time() - since