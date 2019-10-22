

class w2v_api(object):


    def load_word2vec(self, binary=True):
        if self.word_vec_path is None:
            return
        raw_word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.word_vec_path, binary=binary)
        print("load w2v done")
        # clean up lines for memory efficiency
        self.word2vec = []
        oov_cnt = 0
        for v in self.vocab:
            if v not in raw_word2vec:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec - raw_word2vec[v]
            self.word2vec.append(vec)
        print("word2vec cannot cover %f vocab" % (float(oov_cnt ) /len(self.vocab)))
