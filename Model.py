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
                 num_layers=None,
                 hidden_size=None):
        """
        :param encoded_input: list of padded utterance (encoded)
        """
        super(UtteranceEncoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        embeddings_dim = pretrained_embeddings.shape[1]

        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(pretrained_embeddings),
                                                      freeze=freeze_embeddings)

        self.bilstm = nn.LSTM(input_size=embeddings_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True
                              )
        self.bilstm.apply(init_weights)

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, seq_token):
        """
        :param encoded_input: padded encoded sentence
        :return:
        """
        print(' ===== UTTERANCE ENCODER =====')
        input = torch.Tensor(seq_token).long()  # torch.Size([512])

        embed = self.embedding(input)  # shape = torch.Size([1, 512, 50]) batch_size, seq_len, input_size
        embed = embed.unsqueeze(0)
        # print(f'EMBEDDING:\n{embed}')

        hidden_state = self.init_state(1)  # torch.Size([2, 1, 256]) num_directions, batch_size, hidden_size
        # print(f'HIDDEN STATE:\n{hidden_state}')

        output, (hidden, cell) = self.bilstm(embed, hidden_state)  # output torch.Size([1, 512, 512]) batch_size, seq_len, num_directions * hidden_size
        # print(f'OUTPUT:\n{output}')
        # print(f'H_N:\n{hidden}')
        # print(f'C_N:\n{cell}')
        # print("="*50)

        # concat last hidden state of forward LSTM and backward LSTM
        encoder_out = torch.concat((hidden[0], hidden[1]), dim=-1)
        return encoder_out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BiLSTM_Attention(nn.Module):
    """ input: utterance vector dari UtteranceEncoder
    """

    def __init__(self, config):
        super(BiLSTM_Attention, self).__init__()

        pretrained_embeddings = config['pretrained_embeddings']
        freeze_embeddings = config['freeze_embeddings']
        en_hidden_size = config['en_hidden_size']
        output_size = 3
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        SAVE_PATH = 'model_state_dict'
        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)

        self.encoder = UtteranceEncoder(pretrained_embeddings=pretrained_embeddings,
                                        freeze_embeddings=freeze_embeddings,
                                        num_layers=self.num_layers,
                                        hidden_size=en_hidden_size
                                        )

        self.bilstm = nn.LSTM(input_size=2*en_hidden_size, # size output dari encoder (2*seq_len)
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True
                              )
        self.bilstm.apply(init_weights)
        self.fc_attention = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.fc_attention.apply(init_weights)

        self.regression = nn.Linear(in_features=self.hidden_size*2,
                                    out_features=output_size)
        self.regression.apply(init_weights)

        torch.save({'encoder_state_dict': self.encoder.state_dict(),
                    'bilstm_state_dict': self.bilstm.state_dict()},
                   SAVE_PATH+'/model.txt')

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs):
        # input: vector utterance
        inputs = torch.Tensor(inputs)  # seq_len, hidden_size
        print(f'input size:{inputs.size()}')

        # 2x karena output encoder adalah 2*seq_len
        encoder_out = torch.empty(size=(inputs.size()[0], 2*inputs.size()[1]))  # seq_len, hidden_size

        for i in range(len(inputs)):
            encoder_out[i] = self.encoder(inputs[i])
        encoder_out = encoder_out.unsqueeze(0)  # batch_size, seq_len, hidden_size

        # init hidden state
        hidden_state = self.init_state(1)  # seq_len, batch_size, hidden_size

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