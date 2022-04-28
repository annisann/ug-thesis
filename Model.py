import json
import os
import ast
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn


class PretrainedEmbeddings(object):
    """ A wrapper around pre-trained word vectors and their use """

    def load_from_file(self, dim):  # vocab glove
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            dim (str): dimension of GloVe
        Returns:
            instance of PretrainedEmbeddigns
        """
        UNKNOWN_TOKEN = '<UNK>'
        PADDING_TOKEN = '<PAD>'

        self.dim = dim
        self.word2index = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}  # JANGAN PAKE DICT
        self.words = [PADDING_TOKEN, UNKNOWN_TOKEN]
        self.embeddings = np.random.uniform(-0.25, 0.25, (2,50)).tolist()

        embedding_file = 'glove.6B.{}d.txt'.format(self.dim)

        with open(embedding_file, encoding="utf8") as fp:
            for line in fp.readlines():
                line = line.split()
                word = line[0]
                self.words.append(word)
                vec = [float(x) for x in line[1:]]
                self.embeddings.append(vec)
                self.word2index[word] = len(self.word2index)

        return np.array(self.words), np.array(self.embeddings)

    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        return self.embeddings[self.word2index[word]]

    def get_index(self, index):
        index2word = {idx: token for token, idx in self.word2index.items()}

        if index not in index2word:
            raise KeyError(f"Index {index} is not in vocabulary.")
        return index2word[index]

    def __len__(self):
        return len(self.word2index)

    # # TODO: BIKIN BUAT DI DATA BUKAN CLASS EMBEDDINGS
    # def make_embedding_matrix(self, words, max_len):
    #     """
    #     padding and make embedding vectors
    #     :param words: list of tokens
    #     :return:
    #     """
    #     self.word2index, self.embeddings = self.load_from_file(self.dim)
    #     self.embedding_size = self.embeddings.shape[1]
    #     self.final_embeddings = np.zeros((len(words), self.embedding_size))
    #
    #     # PADDING
    #     words += ['<PAD>'] * (max_len - len(words))
    #
    #     for i, word in enumerate(words):
    #         if word in self.word2idx:  # if word is in vocabulary
    #             self.final_embeddings[i, :] = self.embeddings[self.word2idx[word]]
    #         else:
    #             embedding_i = torch.ones(1, self.embedding_size)
    #             torch.nn.init.xavier_uniform_(embedding_i)  # random value, as np.random
    #             self.final_embeddings[i, :] = embedding_i
    #     return self.final_embeddings


class ConvEmoRecogDataset(Dataset):

    def __init__(self, utterance_num, vocab, max_seq_length):
        self.utterance_num = utterance_num

        folder = f'n_{self.utterance_num}'
        scripts = os.listdir(folder)
        self.scripts = [f'{folder}/{script}' for script in scripts if os.path.getsize(f'{folder}/{script}') != 0]

        self.word2idx = {word: index for index, word in enumerate(vocab)}
        self.idx2word = {index: word for word, index in self.word2idx.items()}

        self.padding_token = '<PAD>'
        self.unknown_token = '<UNK>'
        self.max_seq_length = max_seq_length

        self.seq_input = []
        # TODO: seq_inputnya buat per data (train, val, test)

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

                for i in range(len(df)):
                    self.seq_input.append(self.padding(df.iloc[i].token))

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
        seq_tokens.extend([self.padding_token] * (self.max_seq_length-len(seq_tokens)))

        for i in range(len(seq_tokens)):
            if seq_tokens[i] not in self.word2idx:
                seq_tokens[i] = self.word2idx[self.unknown_token]
            else:
                seq_tokens[i] = self.word2idx[seq_tokens[i]]
        return seq_tokens


    # def __len__(self):
    #     return len(self.train) / self.utterance_num, \
    #            len(self.val) / self.utterance_num, \
    #            len(self.test) / self.utterance_num

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


class UtteranceEncoder(nn.Module): # tiap data masuk glove, bilstm, attention, etc.; output: utterance vector
    # inputnya jadi tensor coba
    """
    Compute utterance vector for each utterance
    """
    def __init__(self, config):
        """
        :param encoded_input: list of padded utterance (encoded)
        """
        super(UtteranceEncoder, self).__init__()

        # EMBEDDING
        pretrained_embeddings = config['pretrained_embeddings']
        freeze_embeddings = config['freeze_embeddings']
        # LSTM
        hidden_size = config['hidden_size']
        bidirectional = config['bidirectional']
        num_layers = config['num_layers']

        self.vocab_size = pretrained_embeddings.shape[0]
        self.embeddings_dim = pretrained_embeddings.shape[1]
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pretrained_embeddings),
                                                      freeze=freeze_embeddings)
        self.lstm = nn.LSTM(input_size=self.embeddings_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional
                            )
        # self.attention =

    def forward(self, encoded_input):
        """
        :param encoded_input: padded encoded sentence
        :return:
        """
        encoded_input = torch.Tensor(encoded_input).long()
        embed = self.embedding(encoded_input)
        
