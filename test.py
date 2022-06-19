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

# load hyperparameters from yml file.
with open(f'outputs/run_{arg["n"]}/hyperparameters.yml') as f:
    hyperparam_tuples = re.findall('(\w+): (.+)', f.read(), re.MULTILINE)
    hyperparam_dict = {k: v for k, v in hyperparam_tuples}

non_int_val = ['optimizer', 'with_attention', 'lr_scheduler', 'early_stopping', 'lr', 'embedding_dropout_rate']
is_float_val = ['lr', 'embedding_dropout_rate']

for k, v in hyperparam_dict.items():
    if k not in non_int_val:
        hyperparam_dict[k] = int(v)
    elif k in is_float_val:
        hyperparam_dict[k] = float(v)

# load pretrained embeddings
pretrain = PretrainedEmbeddings()
vocab, embeddings = pretrain.load_from_file(hyperparam_dict['dim'])
dataset = ConvEmoRecogDataset(utterance_num=hyperparam_dict['n_utterances'],
                              vocab=vocab,
                              max_seq_length=hyperparam_dict['max_seq_len'])
traindf, valdf, testdf = dataset.load_dataset()
test_seq = prepare_data(testdf, hyperparam_dict['n_utterances'])

# create new dir
test_dir = create_test_dir()

# load weight from path
model = BiLSTM_Attention(hyperparam_dict, embeddings)
model.load_state_dict(torch.load(f'outputs/run_{arg["n"]}/state_dict.pt')) # kalo mau diautomate, ganti ini.. lot of work? nyampe gak ya lol
# loss function
criterion = nn.MSELoss()

print(f"i\t\tActual VAD\t\t\tPredicted VAD\tLoss")
print("-"*50)

# write test result init
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

        # display true vad value, predicted vad value, and MSE loss
        print(f"{i}\t({v_act:.2f}, {a_act:.2f}, {d_act:.2f})\t({v_pred:.2f}, {a_pred:.2f}, {d_pred:.2f})\t{loss:.2f}")

    # write true vad value, predicted vad value, and MSE loss
    with open(f'{test_dir}/out.txt', "a") as savefile:
        savefile.write(f"{i}\t"
                       f"({v_act:.2f}, {a_act:.2f}, {d_act:.2f})\t"
                       f"({v_pred:.2f}, {a_pred:.2f}, {d_pred:.2f})\t"
                       f"{loss:.2f}\n")

# write the average loss of test data
with open(f'{test_dir}/out.txt', "a") as savefile:
    savefile.write(f"\nTotal Loss: {total_loss/len(test_seq)}")

# show the average loss of test data
print(total_loss/len(test_seq))
