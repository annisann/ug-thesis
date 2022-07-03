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

    def __init__(self,
                 pretrained_embeddings,
                 freeze_embeddings,
                 num_layers,
                 hidden_size,
                 dropout_rate):
        """
        :param encoded_input: list of padded utterance (encoded)
        """
        super(UtteranceEncoder, self).__init__()

        embeddings_dim = pretrained_embeddings.shape[1]
        print(f"Encoder dim {embeddings_dim}")
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout_rate)

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
        :param seq_token: padded encoded sentence
        :return:
        """
        input = torch.Tensor(seq_token).long()  # torch.Size([512])

        embed = self.embedding(input)  # batch_size, seq_len, input_size
        embed = self.dropout(embed)
        embed = embed.unsqueeze(0)

        hidden_state = self.init_state(1)  # num_directions, batch_size, hidden_size

        output, (hidden, cell) = self.bilstm(embed, hidden_state)  # out: batch_size, seq_len, num_directions * hidden_size

        # concat last hidden state of forward LSTM and backward LSTM
        encoder_out = torch.concat((hidden[0], hidden[1]), dim=-1)  # 1, 2*hidden_size
        return encoder_out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class BiLSTM_Attention(nn.Module):
    """ input: utterance vector dari UtteranceEncoder
    """

    def __init__(self, config, pretrained_embeddings):
        super(BiLSTM_Attention, self).__init__()

        self.en_hidden_size = config['en_hidden_size']
        en_n_layer = config['en_n_layer']
        embedding_dropout_rate = config['embedding_dropout_rate']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.with_attention = config['with_attention']
        output_size = 3

        self.encoder = UtteranceEncoder(pretrained_embeddings=pretrained_embeddings,
                                        freeze_embeddings=True,
                                        num_layers=en_n_layer,
                                        hidden_size=self.en_hidden_size,
                                        dropout_rate=embedding_dropout_rate
                                        )

        self.bilstm = nn.LSTM(input_size=2*self.en_hidden_size,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              bidirectional=True,
                              batch_first=True,
                              )

        self.fc_attention = nn.Linear(self.hidden_size*2, self.hidden_size*2,
                                      bias=False)

        self.regression = nn.Linear(in_features=self.hidden_size*2,
                                    out_features=output_size)

        self.bilstm.apply(init_weights)
        self.fc_attention.apply(init_weights)
        self.regression.apply(init_weights)

    def init_state(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size))

    def forward(self, inputs):
        inputs = torch.Tensor(inputs) # seq_len, hidden_size

        # 2x karena output encoder adalah 2*seq_len
        encoder_out = torch.empty(size=(inputs.size()[0], 2*self.en_hidden_size))  # seq_len, hidden_size
        for i in range(len(inputs)):
            encoder_out[i] = self.encoder(inputs[i])
        encoder_out = encoder_out.unsqueeze(0)  # batch_size, seq_len, hidden_size

        # init hidden state
        hidden_state = self.init_state(1)  # seq_len, batch_size, hidden_size

        # BiLSTM
        output, (hidden, cell) = self.bilstm(encoder_out, hidden_state)
        print(f"output shape: {output.size()}")

        if self.with_attention:
            # Bahdanau's Attention
            align_weight = nn.Parameter(torch.FloatTensor(1, self.hidden_size*2))
            x = torch.tanh(self.fc_attention(output))  # batch, seq_len, hidden_size
            # calculate alignment scores
            alignment_scores = x.bmm(align_weight.unsqueeze(2))  # batch, seq_len, 1
            # softmax the alignment scores -> attention weights
            attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)  # batch, seq_len
            # multiply attention weights with output
            context_vector = attn_weights.unsqueeze(0).bmm(output) # 1, batch, seq_len X batch, seq_len, num_dir * hidden = 1, batch, num_dir*hidden
            # print(f"x shape: {x.size()}")
            # print(f"alignment score shape: {alignment_scores.size()}")
            # print(f"attention weight shape: {attn_weights.size()}")
            # print(f"context vector shape: {context_vector.size()}")

            # Regression
            out = self.regression(context_vector).flatten()
            return out
        else:
            output = output.squeeze(0)[-1]
            out = self.regression(output).flatten()
            return out
