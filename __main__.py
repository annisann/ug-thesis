from DataProcessing import *
from Preprocessing import *
from Dataset import *
from Model import *
import glob
import os
import re
import json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import ast
import torch
import time
from tqdm import tqdm


# INITIALIZE CLASS
dp = DataProcessing()
# pp = Preprocessing()

# XXXXXXXXXXXXXXXXXXXX DATA PROCESSING XXXXXXXXXXXXXXXXXXXX
# PATH LIST
scripts_path = [path for path in glob.glob(dp.trans_path)]
emo_path = [path for path in glob.glob(dp.emo_path)]

# DEFINE LISTS OF TRANSCRIPT AND EMO
transcripts = [dp.getTranscription(path) for path in scripts_path]
emos = [dp.getVAD(path) for path in emo_path]

# CONCAT TRANSCRIPT AND ITS EMO LABEL
i_trans = 0
n_scripts = list()
while i_trans < len(transcripts):
    i_utt = 0
    script_n = list()
    while i_utt < len(transcripts[i_trans]):
        currentID = transcripts[i_trans][i_utt]['id']
        if re.match(r'[FM]\d{3}', currentID):
            if [currentID] == list(utt['id'] for utt in transcripts[i_trans] if utt['id'] == currentID):
                index_emo = emos[i_trans].index(next(filter(lambda e: e['id'] == currentID, emos[i_trans])))
                script_n.append(dict(transcripts[i_trans][i_utt].items() | emos[i_trans][index_emo].items()))
        i_utt += 1
    n_scripts.append(script_n)
    i_trans += 1

# FIX UTTERANCES
fixed_scripts = list()
for scripts in n_scripts:
    fixed_scripts.append(dp.fixUtterance(scripts))

# # XXXXXXXXXXXXXXXXXXXX PREPROCESSING XXXXXXXXXXXXXXXXXXXX
# for i, script in enumerate(fixed_scripts):
#     utterances = [uttDict['utterance'] for uttDict in script]
#     expanded = pp.expand(utterances)
#
#     for j in range(len(script)):
#         # EXPAND CONTRACTION
#         script[j]['utterance'] = expanded[j]
#         # CASEFOLDING
#         script[j]['utterance'] = pp.casefolding(script[j]['utterance'])
#         # PUNCTUATION FILTERING
#         script[j]['utterance'] = pp.filterPunct(script[j]['utterance'])
#         # TOKENIZING
#         script[j]['token'] = pp.tokenizing(script[j]['utterance'])
#
#     preprocessed_path = 'preprocessed'
#     print(f'{preprocessed_path}/script_{i}')
#     if not os.path.exists(preprocessed_path):
#         os.mkdir(preprocessed_path)
#
#     with open(f'{preprocessed_path}/script_{i}', 'w+') as file:
#         file.write(json.dumps(script))

# XXXXXXXXXXXXXXXXXXXX SPLIT DATA XXXXXXXXXXXXXXXXXXXX
# TODO: MAKE A LOOP? n_utterances
print(dp.splitData(4, 'preprocessed'))
print(dp.splitData(6, 'preprocessed'))
print(dp.splitData(8, 'preprocessed'))

# XXXXXXXXXXXXXXXXXXXX MODEL XXXXXXXXXXXXXXXXXXXX
# PRETRAINED EMBEDDINGS
dim = 50
pretrain = PretrainedEmbeddings()
vocab, embeddings = pretrain.load_from_file(dim)

# SPLIT DATA (TRAIN, DEV, TEST)
n_utterances = 2 # buat [2, 4, 6, 8]
max_seq_length = 10

config = {
    # ENCODER
    'pretrained_embeddings': embeddings,
    'freeze_embeddings': True,
    'en_hidden_size': max_seq_length,

    # EMO RECOG
    'num_layers': 1,
    'hidden_size': 8
}

model = BiLSTM_Attention(config)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Model has {count_parameters(model):,} trainable parameters')


#max seqlen masuk parameter gak
def prepare_data(n_utterances):
    """
    :param dataset: type of dataset
    :param n_utterances: num of utterances for input
    :return: tuple of [seq_token_i, ..., seq_token_n], mean(V), mean(A), mean(D)
    """
    dataset = ConvEmoRecogDataset(utterance_num=n_utterances,
                                  vocab=vocab,
                                  max_seq_length=max_seq_length)
    traindf, valdf, testdf = dataset.load_dataset()

    train_seq, val_seq, test_seq = [], [], []

    for dataset in [traindf, valdf, testdf]:
        for i in range(len(dataset)):
            if i % n_utterances == 0:
                data = tuple(
                    (np.array(list(dataset.seq[i:i + n_utterances])),
                     torch.from_numpy(np.array(sum(list(dataset.v[i:i + n_utterances])) / n_utterances, dtype=np.float32)),
                     torch.from_numpy(np.array(sum(list(dataset.a[i:i + n_utterances])) / n_utterances, dtype=np.float32)),
                     torch.from_numpy(np.array(sum(list(dataset.d[i:i + n_utterances])) / n_utterances, dtype=np.float32))
                     )
                )
            if dataset == train:
                train_seq.append(data)
            elif dataset == val:
                val_seq.append(data)
            elif dataset == test:
                test_seq.append(data)
    return train_seq, val_seq, test_seq

train_data, val_data, test_data = prepare_data(n_utterances)

data_dict = {'train': train_data,
             'val': val_data,
             'test': test_data}
dataset_sizes = {'train': len(train_data),
                 'val': len(val_data),
                 'test': test_data}


def train(model, criterion, optimizer, num_epochs=5):

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            running_loss = 0.0

            iterator = tqdm(data_dict[phase])

            for input in iterator:
                time.sleep(0.2)
                optimizer.zero_grad()

                input_utterances, v_act, a_act, d_act = input[0], input[1], input[2], input[3]

                v_pred, a_pred, d_pred = model(input_utterances)

                loss_v = criterion(v_pred, v_act)
                loss_a = criterion(a_pred, a_act)
                loss_d = criterion(d_pred, d_act)
                loss = loss_v + loss_a + loss_d

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                iterator.set_postfix_str(f'{phase} loss: {running_loss}')

            epoch_loss = running_loss/dataset_sizes[phase]
            print(f'{phase} loss: {epoch_loss}')
