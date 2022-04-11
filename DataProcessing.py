import glob
import os
import re
import numpy as np


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

    # def loadData(self):
    #     scripts = []
    #
    #     for script in glob.glob(self.trans_path):
    #         scripts.append(script)
    #     # BISA LGSG DIGABUNGIN DI DALEM GLOB
    #     for script in scripts:
    #         rawData = open(script).read()
    #         temp = rawData.split('\n')  # bakal ada "" di list (string kosong)
    #         if "" in temp:
    #             temp.remove("")  # kalo pake .splitlines() gabakal ada string kosong
    #         self.data.append(temp)
    #     return self.data

    def getSpeaker(self, conv):
        # Transcript file
        # if ":" not in conv:
        #     return ""
        speaker = conv.split(":")[0]
        speaker = re.sub(r'^.*_([FM]\w{3}).*$', r'\1', speaker)

        # EmoEvaluation file
        if "\t" in conv:
            speaker = re.sub(r'^.*_([FM]\w{3}).*$', r'\1', speaker)
        return speaker

    def getUtterance(self, conv):
        """
        input: string
        :param conv:
        :return:
        """
        if ":" not in conv:
            return ""
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
                                v=f.split('\t')[3][1:-1].split(',')[0],
                                a=f.split('\t')[3][1:-1].split(',')[1],
                                d=f.split('\t')[3][1:-1].split(',')[2]
                                )
            self.emos.append(self.emoDict)

        return self.emos

    def fixUtterance(self, conv):  # inp: List
        counter = 1
        utt_per_speaker = []

        # COUNT DUPLICATE CONSECUTIVE UTTERANCES
        for i in range(len(conv) - 1):
            currSpeaker = self.getSpeaker(conv[i])  # -> getSpeaker(dictName[i]['speaker'])[0]
            nextSpeaker = self.getSpeaker(conv[i + 1])  # -> getSpeaker(dictName[i+1]['speaker'])[0]

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

            decode = self.getSpeaker(conv[index - 1]) + " : " + ' '.join(  # -> dictName[index-1]['speaker']
                [self.getUtterance(c) for c in conv[index - n:index]])  # -> dictName[index-n:index]['utterance']
            utterances.append(
                decode)  # ke dict baru? fixedDict -> append dictFixed['speaker'] & dictFixed['utterance'] = ' '.join(..)
        self.fixedData.append(utterances)

    def splitData(self, n):
        PATH = os.getcwd() + '/n_' + str(n)
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        idx = [i for i in range(0, len(self.fixedData), n)]
        i = 0
        while i < len(self.fixedData):
            counter = 0
            while counter < len(idx) - 1:
                with open('{}/script{}_{}'.format(PATH, str(i), str(counter)), 'w') as file:
                    if len(self.fixedData[i][idx[counter]:idx[counter + 1]]) == n:
                        for u in self.fixedData[i][idx[counter]:idx[counter + 1]]:
                            file.write('%s\n' % u)
                    else:
                        break
                counter += 1
            i += 1


if __name__ == '__main__':
    dp = DataProcessing()
    # dp.loadData()
    # data = dp.data
    # data = [dp.fixUtterance(d) for d in data]
    # data = dp.fixedData
    # dp.splitData(5)

    # PATH LIST
    scripts_path = [path for path in glob.glob(dp.trans_path)]
    emo_path = [path for path in glob.glob(dp.emo_path)]

    # DEFINE LISTS OF TRANSCRIPT AND  EMO
    transcripts = [dp.getTranscription(path) for path in scripts_path]
    emos = [dp.getVAD(path) for path in emo_path]

    # CONCAT TRANSCRIPT AND ITS EMO LABEL
    i_trans = 0
    transcript_n=list()
    while i_trans < len(transcripts):
        i_utt = 0
        script_n = list()
        while i_utt < len(transcripts[i_trans]):
            currentID = transcripts[i_trans][i_utt]['id'] # F000
            if [currentID] == next([utt['id'] for utt in transcript if utt['id'] == currentID]
                                   for transcript in transcripts):
                index_emo = emos[i_trans].index(next(filter(lambda e: e['id'] == currentID, emos[i_trans])))
                script_n.append(dict(transcripts[i_trans][i_utt].items() | emos[i_trans][index_emo].items()))
            i_utt += 1
        transcript_n.append(script_n)
        i_trans += 1

    # FIX UTTERANCES IN DATA

    # print('len_data: ', len(data))
    # print('range_len_data', range(len(data)))
