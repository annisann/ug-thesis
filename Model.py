import json
import os.path

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class PretrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """

    def load_from_file(self, dim):
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            dim (str): dimension of GloVe
        Returns:
            instance of PretrainedEmbeddigns
        """
        embedding_file = 'glove.6B.{}d.txt'.format(dim)

        self.word2index = {}
        self.embeddings = []

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

    def make_embedding_matrix(self, dim, words):
        self.word2idx, self.embeddings = self.load_from_file(dim)
        self.embedding_size = self.embeddings.shape[1]
        self.final_embeddings = np.zeros((len(words), self.embedding_size))

        for i, word in enumerate(words):
            if word in self.word2idx:
                self.final_embeddings[i, :] = self.embeddings[self.word2idx[word]]
            else:
                embedding_i = torch.ones(1, self.embedding_size)
                torch.nn.init.xavier_uniform_(embedding_i)
                self.final_embeddings[i, :] = embedding_i
        return self.final_embeddings


class ConvEmoRecogDataset(Dataset):
    def load_dataset(self, utterance_num):
        folder = f'n_{utterance_num}/'
        scripts = os.listdir(folder)

        dataframes = []
        # TODO: load data from folder

        for script in scripts:
            if os.path.getsize(folder + script) != 0:
                f = open(folder + script).read()
                f = ast.literal_eval(f)
                data = pd.DataFrame(f)
                dataframes.append(data)

        dataframes = pd.concat(dataframes)
        return dataframes

    def __init__(self, utterance_num):
        self.data = self.load_dataset(utterance_num)
        self.utterance_num = utterance_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # index = file ke-n
        index_1 = index * self.utterance_num
        index_2 = index_1 + self.utterance_num
        data = self.data.loc[index_1:index_2]
        token, v, a, d = data.token, data.v, data.a, data.d

        return np.array(token), np.array(v), np.array(a), np.array(d)
        # TODO: get max_len of token DF.token
