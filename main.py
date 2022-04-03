import glob
import re

class DataProcessing:
    """
    merge utterance from the same speaker.
    """

    def __init__(self):
        self.path = 'dataset/Session*/dialog/transcriptions/*'
        self.data = []

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
        utterances = []

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
        for n in utt_per_speaker:
            index += n

            decode = self.getSpeaker(conv[index - 1]) + " : " + ' '.join(
                [self.getUtterance(c) for c in conv[index - n:index]])
            utterances.append(decode)

        return utterances


if __name__ == '__main__':
    dp = DataProcessing()
    dp.loadData()
    data = dp.data

    _ = []
    for d in data:
        _.append(len(d) % 2)
    print(len(_))
    print(_.count(0))
