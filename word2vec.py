from gensim.models.keyedvectors import KeyedVectors

def load(path, binary=True):
    return Word2Vec.load_word2vec_format(path, binary=binary)


class Word2Vec(KeyedVectors):
    pass
