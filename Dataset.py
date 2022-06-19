from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import ast
import pandas as pd
import numpy as np
import torch


class ConvEmoRecogDataset:
    """
    Load preprocessed data, split, and pad data.
    """

    def __init__(self, utterance_num, vocab, max_seq_length):
        self.utterance_num = utterance_num

        folder = f'{self.utterance_num}-utterances'
        scripts = os.listdir(folder)
        self.scripts = [f'{folder}/{script}' for script in scripts if os.path.getsize(f'{folder}/{script}') != 0]

        self.word2idx = {word: index for index, word in enumerate(vocab)}
        self.idx2word = {index: word for word, index in self.word2idx.items()}

        self.padding_token = '<PAD>'
        self.unknown_token = '<UNK>'
        self.max_seq_length = max_seq_length

    def load_dataset(self):
        # ratio 8:1:1
        trainData, testData = train_test_split(self.scripts, train_size=0.8, random_state=10)
        testData, valData = train_test_split(testData, test_size=0.5, random_state=8)

        traindf, valdf, testdf = [], [], []

        for data in [trainData, valData, testData]:
            for script in data:
                f = open(script).read()
                f = ast.literal_eval(f)
                df = pd.DataFrame(f)

                # add seq token
                df['seq'] = [self.padding(df.iloc[i].token[:self.max_seq_length]) for i in range(len(df))]

                if data == trainData:
                    traindf.append(df)
                elif data == valData:
                    valdf.append(df)
                elif data == testData:
                    testdf.append(df)

        traindf = pd.concat(traindf, ignore_index=True)
        valdf = pd.concat(valdf, ignore_index=True)
        testdf = pd.concat(testdf, ignore_index=True)

        return traindf, valdf, testdf

    def padding(self, tokens):
        """
        :param tokens: list
        :return:
        """
        seq_tokens = tokens.copy()
        seq_tokens.extend([self.padding_token] * (self.max_seq_length - len(seq_tokens)))

        for i in range(len(seq_tokens)):
            if seq_tokens[i] not in self.word2idx:
                seq_tokens[i] = self.word2idx[self.unknown_token]
            else:
                seq_tokens[i] = self.word2idx[seq_tokens[i]]
        return seq_tokens

    def get_data(self, data, index):
        # index = file ke-n
        index_1 = index * self.utterance_num
        index_2 = index_1 + self.utterance_num
        data = data.iloc[index_1:index_2]
        token, v, a, d = data.token, data.v, data.a, data.d

        return np.array(token), np.array(v), np.array(a), np.array(d)

    def max_token_len(self):
        return max(list(map(lambda token: len(token), self.train.token))), \
               max(list(map(lambda token: len(token), self.val.token))), \
               max(list(map(lambda token: len(token), self.test.token)))


class NUtterancesDataset(Dataset):
    """
    Create Dataset dataset ^_^
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterances, v, a, d = self.data[index][0], self.data[index][1], self.data[index][2], self.data[index][3]

        sample = {"utterances": utterances,
                  "v": v,
                  "a": a,
                  "d": d}
        return utterances, v, a, d


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
