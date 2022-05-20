import json
import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

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
        self.embeddings = np.random.uniform(-0.25, 0.25, (2, 50)).tolist()

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


class UtteranceEncoder(nn.Module):
    """
    Compute utterance vector for each utterance
    Pretrained GloVe Embedding -> BiLSTM -> Max Pooling -> Utterance Vector
    """

    # def __init__(self, config):
    def __init__(self,
                 pretrained_embeddings,
                 freeze_embeddings=True,
                 num_layers=1,
                 hidden_size=256):
        """
        :param encoded_input: list of padded utterance (encoded)
        """
        super(UtteranceEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        vocab_size = pretrained_embeddings.shape[0]
        embeddings_dim = pretrained_embeddings.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pretrained_embeddings),
                                                      freeze=freeze_embeddings)

        self.bilstm = nn.LSTM(input_size=embeddings_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True
                              )

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, seq_token):
        """
        :param encoded_input: padded encoded sentence
        :return:
        """
        input = torch.Tensor(seq_token).long()  # torch.Size([512])

        embed = self.embedding(input)  # shape = torch.Size([1, 512, 50]) batch_size, seq_len, input_size
        embed = embed.unsqueeze(0)
        hidden_state = self.init_state(1)  # torch.Size([2, 1, 256]) num_directions, batch_size, hidden_size

        output, (hidden, cell) = self.bilstm(embed, hidden_state)  # output torch.Size([1, 512, 512]) batch_size, seq_len, num_directions * hidden_size

        max_pooling_out = torch.max(output, 1)[0]
        return max_pooling_out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BiLSTM_Attention(nn.Module):
    """ input: utterance vector dari class sebelumnya
    """

    def __init__(self, config):
        super(BiLSTM_Attention, self).__init__()

        pretrained_embeddings = config['pretrained_embeddings']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        freeze_embeddings = config['freeze_embeddings']

        en_hidden_size = config['en_hidden_size']
        output_size = 3

        self.encoder = UtteranceEncoder(pretrained_embeddings=pretrained_embeddings,
                                        freeze_embeddings=freeze_embeddings,
                                        num_layers=self.num_layers,
                                        hidden_size=en_hidden_size
                                        )

        self.bilstm = nn.LSTM(input_size=2 * en_hidden_size,  # ??? inputnya brp? -> output dari encoder
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True
                              )
        self.fc_attention = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc_attention.apply(init_weights)

        self.regression = nn.Linear(in_features=self.hidden_size*2,
                                    out_features=output_size)
        self.regression.apply(init_weights)

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs):
        # input: vector utterance
        inputs = torch.Tensor(inputs)  #[[i_seq_utt1], ... , [i_seq_uttn]] => size [n_utt, seq_len] torch.Size([2, 512])

        encoder_out = torch.empty(size=(inputs.size()[0], inputs.size()[1]))  # torch.Size([2, 512])
        for i in range(len(inputs)):
            encoder_out[i] = self.encoder(inputs[i])
        encoder_out = encoder_out.unsqueeze(0)  # torch.Size([1, 2, 512])

        # init hidden state
        hidden_state = self.init_state(1)  # torch.Size([2, 1, 512]), torch.Size([2, 1, 512])

        # BiLSTM
        output, (hn, cn) = self.bilstm(encoder_out, hidden_state)

        # Bahdanau's Attention
        u_ti = torch.tanh(self.fc_attention(output))  # batch, seq_len, hidden_size
        alignment_scores = u_ti.bmm(nn.Parameter(torch.FloatTensor(1, self.hidden_size)).unsqueeze(2))  # batch, seq_len, 1
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)  # batch, seq_len
        context_vector = attn_weights.unsqueeze(0).bmm(output) # 1, batch, seq_len X batch, seq_len, num_dir * hidden = 1, batch, num_dir*hidden

        # Regression
        out = self.regression(context_vector).flatten()
        return out