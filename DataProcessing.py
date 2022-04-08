import glob
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
        self.path = 'dataset/Session*/dialog/transcriptions/*'
        self.data = list()
        self.fixedData = list()

    def loadData(self):
        path = self.path
        scripts = []

        for script in glob.glob(path):
            scripts.append(script)

        for script in scripts:
            rawData = open(script).read()
            temp = rawData.split('\n')
            if "" in temp:
                temp.remove("")
            self.data.append(temp)
        return self.data

    def getSpeaker(self, conv):
        if ":" not in conv:
            return ""
        speaker = conv.split(":")[0]
        speaker = re.sub(r'^.*_([FM]).*$', r'\1', speaker)
        return speaker

    def getUtterance(self, conv):
        if ":" not in conv:
            return ""
        utterance = conv.split(":")[1]
        return utterance

    def fixUtterance(self, conv):
        counter = 1
        utt_per_speaker = []

        # COUNT DUPLICATE CONSECUTIVE UTTERANCES
        for i in range(len(conv) - 1):
            currSpeaker = self.getSpeaker(conv[i])
            nextSpeaker = self.getSpeaker(conv[i + 1])

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

            decode = self.getSpeaker(conv[index - 1]) + " : " + ' '.join(
                [self.getUtterance(c) for c in conv[index - n:index]])
            utterances.append(decode)
        self.fixedData.append(utterances)

    def splitData(self, n):
        PATH = os.getcwd() + '/n_' + str(n)
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        idx = [i for i in range(0, len(self.fixedData), n)]
        i = 0
        while i < len(self.fixedData):
            counter = 0
            while counter < len(idx)-1:
                with open('{}/script{}_{}'.format(PATH, str(i), str(counter)), 'w') as file:
                    if len(self.fixedData[i][idx[counter]:idx[counter+1]]) == 3: # valuenya 3
                        for u in self.fixedData[i][idx[counter]:idx[counter+1]]:
                            file.write('%s\n' % u)
                            # print('{} idx[{}]:idx[{}]'.format(i, idx[counter], idx[counter+1]))
                    else:
                        break
                counter += 1
            i += 1

        # if len(self.fixedData[0][idx[0]:idx[1]]) == 3:
        #     print(1)
        # print((self.fixedData[0][idx[0]:idx[1]]))


if __name__ == '__main__':
    dp = DataProcessing()
    dp.loadData()
    data = dp.data
    data = [dp.fixUtterance(d) for d in data]
    data = dp.fixedData
    dp.splitData(3) # BELOM CEK KALO N NYA > 3

    a = 0
    b = len(data)  # len data
    c = 3  # berapa banyak mau displit
    #
    # # data[for i in range len(data)] [a:c?]
    # print(data[0][0:3])
    # print(data[0][3:6])
    # print()
    # print(data[1][0:3])
    # print()

    # print('len_data: ', len(data))
    # print('range_len_data', range(len(data)))
