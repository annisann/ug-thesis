from DataProcessing import DataProcessing
from Preprocessing import Preprocessing
import glob
import os
import re


# INITIALIZE CLASS
dp = DataProcessing()

# PATH LIST
scripts_path = [path for path in glob.glob(dp.trans_path)]
emo_path = [path for path in glob.glob(dp.emo_path)]

# DEFINE LISTS OF TRANSCRIPT AND  EMO
transcripts = [dp.getTranscription(path) for path in scripts_path]
emos = [dp.getVAD(path) for path in emo_path]

# CONCAT TRANSCRIPT AND ITS EMO LABEL
i_trans = 0
n_scripts=list()
while i_trans < len(transcripts):
    i_utt = 0
    script_n = list()
    while i_utt < len(transcripts[i_trans]):
        currentID = transcripts[i_trans][i_utt]['id']
        if re.match(r'[FM]\d{3}', currentID):
            if [currentID] == list(utt['id'] for utt in transcripts[i_trans] if utt['id'] == currentID):
                index_emo = emos[i_trans].index(next(filter(lambda e: e['id'] == currentID, emos[i_trans])))
                script_n.append(dict(transcripts[i_trans][i_utt].items() | emos[i_trans][index_emo].items()))
        i_utt += 1
    n_scripts.append(script_n)
    i_trans += 1

# FIX UTTERANCES
fixed_scripts = list()
for scripts in n_scripts:
    fixed_scripts.append(dp.fixUtterance(scripts))