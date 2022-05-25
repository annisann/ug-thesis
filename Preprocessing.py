import os
import re
from pycontractions import Contractions
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class Preprocessing:

    def __init__(self):
        self.gloveFile = 'glove.6B.300d.txt'
        self.w2vFile = self.gloveFile + '.word2vec'

        # Create word2vec file from GloVe
        if self.w2vFile not in os.listdir():
            print("Create W2V from GloVe!")
            glove2word2vec(self.gloveFile, self.w2vFile)
        else:
            print("Found W2V file!")
            kv = KeyedVectors.load_word2vec_format(self.w2vFile, binary=False)

        # Load pycontractions
        print('Loading KV Model . . .')
        self.PYCONTRACTIONS = Contractions(kv_model=kv)

    def expand(self, utterances):
        """
        Expanding contractions.
            e.g. "He loves everything I've done to him.
                  For example, my cooking suited his taste.
                  Bet he'll miss me a lot?"
                 -> "He loves everything I have done to him.
                     For example, my cooking suited his taste.
                     Bet he will miss me a lot?"
        :param utterance: List
        :return: List
        """
        print('0_0 Expandiiiiiiing 0____0')
        return list(self.PYCONTRACTIONS.expand_texts(utterances, precise=True))

    def casefolding(self, utterance):
        """
        Turn a string into lowercase.
            e.g. "He loves everything I have done to him.
                  For example, my cooking suited his taste.
                  Bet he will miss me a lot?"
                 -> "he loves everything i have done to him.
                     for example, my cooking suited his taste.
                     bet he will miss me a lot?"
        :param utterance: String
        :return: String
        """
        return utterance.lower()

    def filterPunct(self, utterance):
        """
        Filter punctuation in a string.
            e.g. "he loves everything i have done to him.
                  for example, my cooking suited his taste.
                  bet he will miss me a lot?"
                 -> "he loves everything i have done to him
                     for example my cooking suited his taste
                     bet he will miss me a lot"
        :param utterance: String
        :return: String
        """
        return re.sub(r"[^\w\s]", "", utterance)

    def tokenizing(self, utterance):
        """
        Turn a string into words or tokens.
            e.g. "he loves everything i have done to him
                  for example my cooking suited his taste
                  bet he will miss me a lot"
                 -> ["he", "loves", "everything", "i", "have", "done", "to", "him",
                     "for", "example", "my", "cooking", "suited", "his", "taste",
                     "bet", "he", "will", "miss", "me", "a", "lot"]
        :param utterance: String
        :return: tokens: List
        """
        return utterance.split()
