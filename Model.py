import json
import os
import ast
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

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

        embeddings_dim = pretrained_embeddings.shape[1]
        self.num_layers = num_layers
        self.hidden_size = hidden_size

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
        # print(' ===== UTTERANCE ENCODER =====')
        input = torch.Tensor(seq_token).long()  # torch.Size([512])

        embed = self.embedding(input)  # shape = torch.Size([1, 512, 50]) batch_size, seq_len, input_size
        embed = embed.unsqueeze(0)
        # print(f'EMBEDDING:\n{embed}')

        hidden_state = self.init_state(1)  # num_directions, batch_size, hidden_size
        # print(f'HIDDEN STATE:\n{hidden_state}')

        output, (hidden, cell) = self.bilstm(embed, hidden_state)  # out: batch_size, seq_len, num_directions * hidden_size
        # print(f'OUTPUT:\n{output}')
        # print(f'H_N:\n{hidden}')
        # print(f'C_N:\n{cell}')
        # print("="*50)

        # concat last hidden state of forward LSTM and backward LSTM
        encoder_out = torch.concat((hidden[0], hidden[1]), dim=-1) # 1, 2*hidden_size
        return encoder_out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


class BiLSTM_Attention(nn.Module):
    """ input: utterance vector dari UtteranceEncoder
    """

    def __init__(self, config):
        super(BiLSTM_Attention, self).__init__()

        pretrained_embeddings = config['pretrained_embeddings']
        freeze_embeddings = config['freeze_embeddings']
        self.en_hidden_size = config['en_hidden_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        output_size = 3

        self.encoder = UtteranceEncoder(pretrained_embeddings=pretrained_embeddings,
                                        freeze_embeddings=freeze_embeddings,
                                        num_layers=self.num_layers,
                                        hidden_size=self.en_hidden_size
                                        )

        self.bilstm = nn.LSTM(input_size=2*self.en_hidden_size, # size output dari encoder (2*seq_len)
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True
                              )
        self.bilstm.apply(init_weights)
        self.fc_attention = nn.Linear(self.hidden_size*2, self.hidden_size*2,
                                      bias=False)
        self.fc_attention.apply(init_weights)

        self.regression = nn.Linear(in_features=self.hidden_size*2,
                                    out_features=output_size)
        self.regression.apply(init_weights)

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs):
    # def forward(self, input1, input2):
    #     input: vector utterance
        inputs = torch.Tensor(inputs)  # seq_len, hidden_size
        # input1 = torch.Tensor(input1)
        # input2 = torch.Tensor(input2)
        # encoder_out1 = self.encoder(input1)
        # encoder_out2 = self.encoder(input2)
        # encoder_out = torch.concat((encoder_out1, encoder_out2), dim=0)

        # 2x karena output encoder adalah 2*seq_len
        encoder_out = torch.empty(size=(inputs.size()[0], 2*self.en_hidden_size))  # seq_len, hidden_size
        for i in range(len(inputs)):
            encoder_out[i] = self.encoder(inputs[i])
        encoder_out = encoder_out.unsqueeze(0)  # batch_size, seq_len, hidden_size

        # init hidden state
        hidden_state = self.init_state(1)  # seq_len, batch_size, hidden_size

        # BiLSTM
        output, (hidden, cell) = self.bilstm(encoder_out, hidden_state)
        # print(f'OUTPUT:\n{output}')
        # print(f'H_N:\n{hidden}')
        # print(f'C_N:\n{cell}')
        # print("="*50)

        # Bahdanau's Attention
        weight = nn.Parameter(torch.FloatTensor(1, self.hidden_size*2))
        # print(f'w attention:{weight}')
        x = torch.tanh(self.fc_attention(output))  # batch, seq_len, hidden_size
        # print(f'x:{x}')
            # calculate alignment scores
        alignment_scores = x.bmm(weight.unsqueeze(2))  # batch, seq_len, 1
            # softmax the alignment scores -> attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)  # batch, seq_len
            # multiply attention weights with output
        context_vector = attn_weights.unsqueeze(0).bmm(output) # 1, batch, seq_len X batch, seq_len, num_dir * hidden = 1, batch, num_dir*hidden
        # print(f'ALIGNMENT_SCORES:\n{alignment_scores}\n'
        #       f'ATTENTION WEIGHTS:\n{attn_weights}\n'
        #       f'CONTEXT_VECTOR:\n{context_vector}\n')
        # print(f'context vector shape: {context_vector.size()}')

        # Regression
        out = self.regression(context_vector).flatten()
        return out