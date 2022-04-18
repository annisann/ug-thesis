from DataProcessing import DataProcessing
from Preprocessing import Preprocessing
from Model import PretrainedEmbeddings
import glob
import os
import re
import json


# INITIALIZE CLASS
dp = DataProcessing()
pp = Preprocessing()

# PATH LIST
scripts_path = [path for path in glob.glob(dp.trans_path)]
emo_path = [path for path in glob.glob(dp.emo_path)]

# DEFINE LISTS OF TRANSCRIPT AND  EMO
transcripts = [dp.getTranscription(path) for path in scripts_path]
emos = [dp.getVAD(path) for path in emo_path]

# CONCAT TRANSCRIPT AND ITS EMO LABEL
i_trans = 0
n_scripts = list()
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

# SPLIT DATA INTO N-UTTERANCES
N = [2, 4, 6]
for n in N:
    dp.splitData(n, fixed_scripts)

# PREPROCESSING
pathList = ['n_2/*', 'n_4/*', 'n_6/*']
for path in pathList:
    for script_path in glob.glob(path):
        if os.path.getsize(script_path) != 0:
            file = open(script_path, 'r+')
            listOfNUtterances = json.load(file)

            # EXPAND CONTRACTIONS -> input: List of utterances
            utterances = [uttDict['utterance'] for uttDict in listOfNUtterances]
            expanded = pp.expandContractions(utterances)

            for i in range(len(listOfNUtterances)):
                listOfNUtterances[i]['utterance'] = expanded[i]
                # CASEFOLDING
                listOfNUtterances[i]['utterance'] = pp.casefolding(listOfNUtterances[i]['utterance'])
                # PUNCTUATION FILTERING
                listOfNUtterances[i]['utterance'] = pp.filterPunct(listOfNUtterances[i]['utterance'])
                # TOKENIZING
                listOfNUtterances[i]['token'] = pp.tokenizing(listOfNUtterances[i]['utterance'])

            print('{} has preprocessed'.format(script_path))
            file.seek(0)
            file.write(json.dumps(listOfNUtterances))
            file.truncate()

# SPLIT DATA (TRAIN, DEV, TEST)
from sklearn.model_selection import train_test_split

# train: 0.6; val: 0.2; test: 0.2
trainData, testData = train_test_split(os.listdir('n_2'), train_size=0.6, random_state=10)
testData, valData = train_test_split(testData, test_size=0.2, random_state=8)