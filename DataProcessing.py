import glob
import json
import os
import re
import ast


class DataProcessing:
    """
    Processing IEMOCAP raw dataset:
    1) merge transcription and dimensional data.
    2) merge utterance from the same speaker.
    3) split data into n-utterances.
    """

    def __init__(self):
        """
        data = [[Utterance1, Utterance2, ..., UtteranceN],
                [Utterance1, Utterance2, ..., UtteranceN],
                ...
                [Utterance1, Utterance2, ..., UtteranceN]
                ]
        """

        self.trans_path = 'dataset/Session*/dialog/transcriptions/*'
        self.emo_path = 'dataset/Session*/dialog/EmoEvaluation/*.txt'

    def getSpeaker(self, conv):
        """
        :param conv: String
        :return: speaker: String, e.g 'F000'
        """
        speaker = conv.split(":")[0]
        speaker = re.sub(r'^.*_([FM]\w{3}).*$', r'\1', speaker)

        # EmoEvaluation file
        if "\t" in conv:
            speaker = re.sub(r'^.*_([FM]\w{3}).*$', r'\1', speaker)
        return speaker

    def getUtterance(self, conv):
        """
        :param conv: String
        :return: utterance: String, e.g 'Excuse me.'
        """
        utterance = conv.split(":")[1]
        return utterance

    def getTranscription(self, path):
        """
        turn every utterances in file into dictionary, consists of speaker ID and its utterance.
        :param path: transcription file (.txt)
        :return: list of dictionaries
        """
        transcriptions = list()

        file = open(path, 'r').read().splitlines()  # [utt_1, utt_2, .. , utt_n]

        for f in file:
            transcriptDict = dict(id=self.getSpeaker(f),
                                  utterance=self.getUtterance(f)
                                  )
            transcriptions.append(transcriptDict)
        return transcriptions

    def getVAD(self, path):
        """
        get dimensional emotion (valence, arousal, dominance) from EmoEvaluation file.
        :param path: EmoEvaluation file (.txt)
        :return: list of dictionaries
        """
        emos = list()

        file = open(path, 'r').read().splitlines()
        file = [f for f in filter(lambda x: x.startswith('['), file)]

        for f in file:
            emoDict = dict(id=self.getSpeaker(f),
                           v=float(f.split('\t')[3][1:-1].split(',')[0]),
                           a=float(f.split('\t')[3][1:-1].split(',')[1]),
                           d=float(f.split('\t')[3][1:-1].split(',')[2])
                           )
            emos.append(emoDict)

        return emos

    def fixUtterance(self, listOfDict):
        """
        concatenate utterances from the same speaker and its dimensional emotion value.
        :param conv: list of dictionaries [{},{},...,{}]
        :return: list of dictionaries
        """
        counter = 1
        utt_per_speaker = []

        # COUNT CONSECUTIVE DUPLICATE SPEAKERS
        for i in range(len(listOfDict) - 1):
            currSpeaker = listOfDict[i]['id'][0]
            nextSpeaker = listOfDict[i+1]['id'][0]

            if currSpeaker == nextSpeaker:
                counter += 1
            else:
                utt_per_speaker.append(counter)
                counter = 1
        utt_per_speaker.append(counter)

        # MERGE DUPLICATE SPEAKER'S UTTERANCE
        index = 0
        utterances = []
        for n in utt_per_speaker:
            index += n

            fixed = dict(id=listOfDict[index-1]['id'],
                         utterance=' '.join([''.join(listOfDict[i]['utterance']) for i in range(index-n, index)]),
                         v=sum([listOfDict[i]['v'] for i in range(index-n, index)])/n,
                         a=sum([listOfDict[i]['a'] for i in range(index-n, index)])/n,
                         d=sum([listOfDict[i]['d'] for i in range(index-n, index)])/n)
            utterances.append(fixed)
        return utterances

    def splitData(self, n, path):
        """
        split data into n-utterances and save to new dir
        :param n: number of utterances
        :param path: path of preprocessed scripts
        :return:
        """
        FOLDER_NAME = f'{str(n)}-utterances'

        if not os.path.exists(FOLDER_NAME):
            os.mkdir(FOLDER_NAME)

        # for every scripts in preprocessed_data
        for script in os.listdir(path):
            print(f'{path}/{script}')

            # open file, isi file jadi list
            f = open(f'{path}/{script}').read()
            f = ast.literal_eval(f)

            # count length of [] scripts
            script_len = len(f)

            # index[] -> index of 0:2, 2:4, ... -> (0, len_scripts, n)
            idx = [i for i in range(0, script_len+1, n)]

            # while i < len_scripts
            i = 0
            while i < script_len:
                counter = 0
                # while counter < len(index)-1
                while counter < len(idx)-1:
                    # n_utterances -> scripts[i][idx[counter]:idx[counter+1]]
                    n_utterances = f[idx[counter]:idx[counter+1]]
                    # open new file -> nama baru .txt
                    with open(f'{FOLDER_NAME}/{script}_{counter}.txt', 'w') as file:
                        # write file of [{},..,{}] ke path baru
                        file.write(json.dumps(n_utterances))
                    counter+=1
                i+=1