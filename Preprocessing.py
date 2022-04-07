import os

from pycontractions import Contractions
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import re
import nltk

class Preprocessing:

    def __init__(self):
        self.gloveFile = 'glove.6B.300d.txt'
        self.w2vFile = self.gloveFile + '.word2vec'

    def expandContractions(self, utterances):
        """
        input: list
        :return: list
        """
        self.utterances = utterances

        if self.w2vFile not in os.listdir():
            glove2word2vec(self.gloveFile, self.w2vFile)
        else:
            kv = KeyedVectors.load_word2vec_format(self.w2vFile, binary=False)

        # load pycontractions
        cont = Contractions(kv_model=kv)
        cont.load_models()

        return list(cont.expand_texts(self.utterances, precise=True))

    def casefolding(self, utterance):
        self.utterance = utterance
        return self.utterance.lower()

    def filterPunct(self, utterance):
        self.utterance = utterance
        return re.sub(r"[^\w\s]", "", self.utterance)

    def tokenizing(self, utterance):
        self.utterance = utterance
        return self.utterance.split()

if __name__ == '__main__':
    # utterance = ["I'd like to know how I'd done that!",
    #                         "We're going to the zoo and I don't think I'll be home for dinner.",
    #                         "Theyre going to the zoo and she'll be home for dinner."]
    pp = Preprocessing()
    utterances= ["i'd like to know how i'd done that", "ok?????"]
    expanded = pp.expandContractions(utterances)
    casefolded = [pp.casefolding(_) for _ in expanded]
    filtered = [pp.filterPunct(_) for _ in casefolded]
    token = [pp.tokenizing(_) for _ in filtered]