import glob
import json
import os
import re


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
        self.path = path
        self.transcriptions = list()

        file = open(self.path, 'r').read().splitlines()  # [utt_1, utt_2, .. , utt_n]

        for f in file:
            self.transcriptDict = dict(id=self.getSpeaker(f),
                                       utterance=self.getUtterance(f)
                                       )
            self.transcriptions.append(self.transcriptDict)
        return self.transcriptions

    def getVAD(self, path):
        """
        get dimensional emotion (valence, arousal, dominance) from EmoEvaluation file.
        :param path: EmoEvaluation file (.txt)
        :return: list of dictionaries
        """
        self.path = path
        self.emos = list()

        file = open(self.path, 'r').read().splitlines()
        file = [f for f in filter(lambda x: x.startswith('['), file)]

        for f in file:
            self.emoDict = dict(id=self.getSpeaker(f),
                                v=float(f.split('\t')[3][1:-1].split(',')[0]),
                                a=float(f.split('\t')[3][1:-1].split(',')[1]),
                                d=float(f.split('\t')[3][1:-1].split(',')[2])
                                )
            self.emos.append(self.emoDict)

        return self.emos

    def fixUtterance(self, listOfDict):
        """
        concatenate utterances from the same speaker and its dimensional emotion value.
        :param conv: list of dictionaries [{},{},...,{}]
        :return: list of dictionaries
        """
        counter = 1
        utt_per_speaker = []

        # COUNT DUPLICATE CONSECUTIVE UTTERANCES
        for i in range(len(listOfDict) - 1):
            currSpeaker = listOfDict[i]['id'][0]
            nextSpeaker = listOfDict[i+1]['id'][0]

            if currSpeaker == nextSpeaker:
                counter += 1
            else:
                utt_per_speaker.append(counter)
                counter = 1
        utt_per_speaker.append(counter)

        # MERGE DUPLICATE CONS. UTTERANCES INTO ONE
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
        # for every scripts in preprocessed_data
            # open file, isi file jadi list
            # count length of [] scripts
            # index[] -> index of 0:2, 2:4, ... -> (0, len_scripts, n)
            # while i < len_scripts
                # while counter < len(index)-1
                    # n_utterances -> scripts[i][idx[counter]:idx[counter+1]]
                    # open new file -> nama baru script_i_counter.txt
                        # write file of [{},..,{}] ke path baru
                    # counter+=1
                # i+=1

        FOLDER_NAME = f'{str(n)}-utterances'

        if not os.path.exists(FOLDER_NAME):
            os.makedirs(FOLDER_NAME)

        len_data = len(os.listdir(path))  # 151
        idx = [i for i in range(0, len_data, n)]  # ???? len isi scriptnya yaaa

        i = 0
        while i < len_data:
            counter = 0
            while counter < len(idx)-1:
                # with open('{}/script{}_{}.txt'.format(PATH, str(i), str(counter)), 'w') as file:
                with open(f'{FOLDER_NAME}/script_{str(i)}_{str(counter)}.txt', 'w') as file:
                    # n_utterances = scripts[i][idx[counter]:idx[counter + 1]] #?????? ganti kyny
                    if len(n_utterances) == n:
                        file.write(json.dumps(n_utterances))
                    else:
                        break
                counter += 1
            i += 1