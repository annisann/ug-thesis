import numpy as np

class PretrainedEmbeddings:
    """ A wrapper around pre-trained word vectors and their use """

    def load_from_file(self, dim):
        """Instantiate from pre-trained vector file.

        Vector file should be of the format:
            word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
            word1 x1_0 x1_1 x1_2 x1_3 ... x1_N

        Args:
            dim (str): dimension of GloVe
        Returns:
            instance of PretrainedEmbeddigns
        """
        UNKNOWN_TOKEN = '<UNK>'
        PADDING_TOKEN = '<PAD>'

        self.word2index = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
        self.words = [PADDING_TOKEN, UNKNOWN_TOKEN]
        self.embeddings = np.random.uniform(-0.25, 0.25, (2, dim)).tolist()

        embedding_file = f"glove.6B.{dim}d.txt"

        with open(embedding_file, encoding="utf8") as fp:
            for line in fp.readlines():
                line = line.split()
                word = line[0]
                self.words.append(word)

                vec = [float(x) for x in line[1:]]
                self.embeddings.append(vec)
                self.word2index[word] = len(self.word2index)

        return np.array(self.words), np.array(self.embeddings)

    def get_embedding(self, word):
        """
        Args:
            word (str)
        Returns
            an embedding (numpy.ndarray)
        """
        return self.embeddings[self.word2index[word]]

    def get_index(self, index):
        index2word = {idx: token for token, idx in self.word2index.items()}

        if index not in index2word:
            raise KeyError(f"Index {index} is not in vocabulary.")
        return index2word[index]

    def __len__(self):
        return len(self.word2index)