import json
import os
import ast
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class PretrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """

    def load_from_file(self, dim): # vocab glove
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            dim (str): dimension of GloVe
        Returns:
            instance of PretrainedEmbeddigns
        """
        self.dim = dim
        self.word2index = {} # vocabulary
        self.embeddings = []

        self.word2index['<PAD>'] = 0
        self.word2index['<UNK>'] = 1

        embedding_file = 'glove.6B.{}d.txt'.format(self.dim)

        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])

                self.word2index[word] = len(self.word2index)
                self.embeddings.append(vec)

        return self.word2index, np.stack(self.embeddings)

    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        return self.embeddings[self.word2index[word]]

    def get_index(self, index):
        self.index2word = {idx: token for token, idx in self.word2index.items()}

        if index not in self.idx2word:
            raise KeyError(f"Index {index} is not in vocabulary.")
        return self.idx2word[index]

    def __len__(self):
        return len(self.word2index)

    def make_embedding_matrix(self, words, max_len):
        """
        padding and make embedding vectors
        :param words: list of tokens
        :return:
        """

        self.word2index, self.embeddings = self.load_from_file(self.dim)
        self.embedding_size = self.embeddings.shape[1]
        self.final_embeddings = np.zeros((len(words), self.embedding_size))

        # PADDING
        words += ['<PAD>'] * (max_len - len(words))

        for i, word in enumerate(words):
            if word in self.word2idx: # if word is in vocabulary
                self.final_embeddings[i, :] = self.embeddings[self.word2idx[word]]
            else:
                embedding_i = torch.ones(1, self.embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i) # random value, as np.random
                self.final_embeddings[i, :] = embedding_i
        return self.final_embeddings


class ConvEmoRecogDataset(Dataset):

    def load_dataset(self):
        folder = f'n_{self.utterance_num}'
        scripts = os.listdir(folder)
        scripts = [f'{folder}/{script}' for script in scripts if os.path.getsize(f'{folder}/{script}') != 0]

        # SPLIT DATA, hasilnya adalah list of dir ['n_2/..', ..., ...]
        # ratio 8:1:1
        trainData, testData = train_test_split(scripts, train_size=0.8, random_state=10)
        testData, valData = train_test_split(testData, test_size=0.5, random_state=8)

        traindf, valdf, testdf = [], [], []

        for data in [trainData, valData, testData]:
            for script in data:
                f = open(script).read()
                f = ast.literal_eval(f)
                df = pd.DataFrame(f)
                if data == trainData:
                    traindf.append(df)
                elif data == valData:
                    valdf.append(df)
                elif data == testData:
                    testdf.append(df)

        traindf= pd.concat(traindf, ignore_index=True)
        valdf = pd.concat(valdf, ignore_index=True)
        testdf = pd.concat(testdf, ignore_index=True)
        return traindf, valdf, testdf

    def __init__(self, utterance_num):
        self.utterance_num = utterance_num
        self.traindata, self.valdata, self.testdata = self.load_dataset()

    def __len__(self):
        return len(self.traindata)/self.utterance_num,\
               len(self.valdata)/self.utterance_num,\
               len(self.testdata)/self.utterance_num

    def __getdata__(self, data, index):
        # index = file ke-n
        index_1 = index * self.utterance_num
        index_2 = index_1 + self.utterance_num
        data = data.iloc[index_1:index_2]
        token, v, a, d = data.token, data.v, data.a, data.d

        return np.array(token), np.array(v), np.array(a), np.array(d)

    def max_token_len(self):
        return max(list(map(lambda token: len(token), self.traindata.token))),\
               max(list(map(lambda token: len(token), self.valdata.token))),\
               max(list(map(lambda token: len(token), self.testdata.token)))
