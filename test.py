import torch
import re
from Model import *
from Dataset import *
from PretrainedEmbeddings import *
from utils import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-dir', dest='n', type=int)
arg = vars(parser.parse_args())

arg['n'] = 27


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

# load model, isi paramnya dari .yaml
with open(f'outputs/run_{n}/hyperparameters.yml') as f:
    hyperparam_tuples = re.findall('(\w+): (.+)', f.read(), re.MULTILINE)
    hyperparam_dict = {k: v for k, v in hyperparam_tuples}

non_int_val = ['optimizer', 'with_attention', 'lr_scheduler', 'early_stopping', 'lr', 'embedding_dropout_rate']
is_float_val = ['lr', 'embedding_dropout_rate']

for k, v in hyperparam_dict.items():
    if k not in non_int_val:
        hyperparam_dict[k] = int(v)
    elif k in is_float_val:
        hyperparam_dict[k] = float(v)

pretrain = PretrainedEmbeddings()

vocab, embeddings = pretrain.load_from_file(hyperparam_dict['dim'])

dataset = ConvEmoRecogDataset(utterance_num=hyperparam_dict['n_utterances'],
                              vocab=vocab,
                              max_seq_length=hyperparam_dict['max_seq_len'])
traindf, valdf, testdf = dataset.load_dataset()
test_seq = prepare_data(testdf, hyperparam_dict['n_utterances'])

test_dir = create_test_dir()

# load weight from path
model = BiLSTM_Attention(hyperparam_dict, embeddings)
model.load_state_dict(torch.load(f'outputs/run_{n}/state_dict.pt')) # kalo mau diautomate, ganti ini.. lot of work? nyampe gak ya lol

criterion = nn.MSELoss()

print(f"i\t\tActual VAD\t\t\tPredicted VAD\tLoss")
print("-"*50)

with open(f'{test_dir}/out.txt', "w") as savefile:
    savefile.write(f"i\t\tActual VAD\t\t\tPredicted VAD\tLoss\n")

total_loss = 0.
for i, input in enumerate(test_seq):
    with torch.no_grad():
        input_utterances, v_act, a_act, d_act = input[0], input[1], input[2], input[3]

        v_pred, a_pred, d_pred = model(input_utterances)
        loss_v = criterion(v_pred, v_act)
        loss_a = criterion(a_pred, a_act)
        loss_d = criterion(d_pred, d_act)
        loss = loss_v + loss_a + loss_d
        total_loss += loss

        print(f"{i}\t({v_act:.2f}, {a_act:.2f}, {d_act:.2f})\t({v_pred:.2f}, {a_pred:.2f}, {d_pred:.2f})\t{loss:.2f}")

    with open(f'{test_dir}/out.txt', "a") as savefile:
        savefile.write(f"{i}\t"
                       f"({v_act:.2f}, {a_act:.2f}, {d_act:.2f})\t"
                       f"({v_pred:.2f}, {a_pred:.2f}, {d_pred:.2f})\t"
                       f"{loss:.2f}\n")

with open(f'{test_dir}/out.txt', "a") as savefile:
    savefile.write(f"\nTotal Loss: {total_loss/len(test_seq)}")

print(total_loss/len(test_seq))
