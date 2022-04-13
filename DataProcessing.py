import glob
import json
import os
import re


class DataProcessing:
    """
    merge utterance from the same speaker.
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
        self.data = list()
        self.fixedData = list()

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
        return self.transcriptions  # transcriptionsnya [{}, .. , {}], bukan [{},..,{}],[{},...,{}],...]

    def getVAD(self, path):
        ''''''
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

    def fixUtterance(self, listOfDict):  # inp: List
        """
        :param conv: list of dictionaries [{},{},...,{}]
        :return:
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
                         # id=self.getSpeaker(listOfDict[index-1]['id']),
                         # utterances=' '.join([self.getUtterance(c) for c in listOfDict[index-n:index]['utterance']]),
                         utterances = ' '.join([''.join(listOfDict[i]['utterance']) for i in range(index-n, index)]),
                         v=sum([listOfDict[i]['v'] for i in range(index-n, index)])/n,
                         a=sum([listOfDict[i]['a'] for i in range(index-n, index)])/n,
                         d=sum([listOfDict[i]['d'] for i in range(index-n, index)])/n)
            utterances.append(fixed)
        self.fixedData = utterances
        return utterances

    def splitData(self, n, scripts):
        """
        split data into n-utterances.
        :param n: number of utterances wanted
        :param scripts: list of list of dictionaries [[utt_1, utt_2, ... , utt_n], ... , [utt_1, ... , utt_n]]
        :return: -
        """
        PATH = os.getcwd() + '/n_' + str(n)
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        idx = [i for i in range(0, len(scripts), n)]
        i = 0
        while i < len(scripts):
            counter = 0
            while counter < len(idx)-1:
                with open('{}/script{}_{}.txt'.format(PATH, str(i), str(counter)), 'w') as file:
                    n_utterances = scripts[i][idx[counter]:idx[counter + 1]]
                    if len(n_utterances) == n:
                        file.write(json.dumps(n_utterances))
                    else:
                        break
                counter += 1
            i += 1