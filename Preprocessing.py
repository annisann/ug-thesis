import os
import re
from pycontractions import Contractions
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class Preprocessing:

    def __init__(self):
        self.gloveFile = 'glove.6B.300d.txt'
        self.w2vFile = self.gloveFile + '.word2vec'

        if self.w2vFile not in os.listdir():
            print("Create W2V from GloVe!")
            glove2word2vec(self.gloveFile, self.w2vFile)
        else:
            print("Found W2V file!")
            kv = KeyedVectors.load_word2vec_format(self.w2vFile, binary=False)

        # load pycontractions
        print('Loading KV Model . . .')
        self.PYCONTRACTIONS = Contractions(kv_model=kv)

    def expand(self, utterance):
        print('0_0 Expandiiiiiiing 0____0')
        return list(self.PYCONTRACTIONS.expand_texts(utterance, precise=True))

    def casefolding(self, utterance):
        return utterance.lower()

    def filterPunct(self, utterance):
        return re.sub(r"[^\w\s]", "", utterance)

    def tokenizing(self, utterance):
        return utterance.split()