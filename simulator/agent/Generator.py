from __future__ import division, print_function, absolute_import
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pkl
import numpy as np

from simulator.language_model.counter import build_vocabulary, count_ngrams
from simulator.language_model.ngram import MLENgramModel
#from simulator.language_model.util import read_pickle

from simulator.agent.tokenizer import detokenize
from config import Config




#todo lexicalize and clean each utterance before creating tfidf
class Generator(object):
    def __init__(self, vec_dir=None, persuader=1):
        # self.config = config
        self.df = pd.read_csv(Config.csv_for_generator)
        # self.history_utt_name = config.history_utt_name
        # self.history_label_name = config.history_label_name

        self.persuader = persuader
        # if self.persuader:
        #     self.df = self.df[self.df['B4'] == 0]
        # else:
        #     self.df = self.df[self.df['B4'] == 1]
        if vec_dir:
            with open(vec_dir, "r") as fh:
                self.vectorizer = pkl.load(fh)
        else:
            self.vectorizer1 = TfidfVectorizer()
            self.vectorizer2 = TfidfVectorizer()
            self.build_tfidf()

    def build_tfidf(self):
        print("building tfidf in generator")
        context1 = self.df['context']
        context2 = self.df['context']

        context_lower1 = [c.lower() for c in context1]
        context_lower2 = [c.lower() for c in context2]
        self.tfidf_matrix1 = self.vectorizer1.fit_transform(context_lower1)
        self.tfidf_matrix2 = self.vectorizer2.fit_transform(context_lower2)

    def _add_filter(self, locs, cond):
        locs.append(locs[-1] & cond)

    def _select_filter(self, locs):
        print([np.sum(loc) for loc in locs])
        for loc in locs[::-1]:
            if np.sum(loc) > 0:
                return loc
        return locs[0]

    def get_filter(self, act_param, used_templates=None):
        if used_templates:
            loc = (~self.df.index.isin(used_templates))
            available_df = self.df[loc]
            condition = (available_df[Config.history_label_name].isin(label_context)) & (available_df[Config.cur_label_name].isin(label_utt))
            available_loc = available_df[condition].index
            available_loc = self.df.index.isin(available_loc)

            if np.sum(available_loc) > 0:
                return available_loc
            else:
                return None
        else:
            condition = (self.df['act_param'].isin(act_param))
            available_loc = self.df[condition].index
            available_loc = self.df.index.isin(available_loc)

            if np.sum(available_loc) > 0:
                return available_loc
            else:
                return None

    def retrieve(self, context, act_param, used_templates=None, topk=Config.topk, T=1.):
        if isinstance(act_param, str):
            act_param = [act_param]
        # if isinstance(label_utt, str):
        #     label_utt = [label_utt]
        if Config.generator_debug:
            print("act_param: {}".format(act_param))
            # print("label_utt: {}".format(label_utt))
        loc = self.get_filter(act_param=act_param, used_templates=used_templates)
        if loc is None:
            return None, None

        context = context.lower()
        if isinstance(context, list):
            context = detokenize(context)
        features1 = self.vectorizer1.transform([context])
        scores1 = self.tfidf_matrix1 * features1.T
        scores1 = scores1.todense()[loc]
        scores1 = np.squeeze(np.array(scores1), axis=1)

        features2 = self.vectorizer2.transform([context])
        scores2 = self.tfidf_matrix2 * features2.T
        scores2 = scores2.todense()[loc]
        scores2 = np.squeeze(np.array(scores2), axis=1)
        assert scores1.shape == scores2.shape

        scores = np.hstack([scores1, scores2])
        ids = np.argsort(scores)[::-1][:topk]
        ids_converted = []
        for one_id in ids:
            if one_id < scores1.shape[0]:
                ids_converted.append(one_id)
                # print("use last sent!")
            else:
                ids_converted.append(one_id-scores1.shape[0])
                # print("use all sents")
        ids_converted = [one_id if one_id < scores1.shape[0] else (one_id-scores1.shape[0]) for one_id in ids]
        topk_scores = np.sort(scores)[::-1][:topk]

        candidates = self.df[loc]
        # print("candidate size: {}".format(candidates.shape))
        candidates = candidates.iloc[ids_converted]['utt']#.values
        rows = self.df[loc]
        rows = rows.iloc[ids_converted]
        # logp = rows['logp'].values

        return self.sample(topk_scores, candidates, topk_scores, T)

    def sample(self, scores, templates, tfidf_scores, T=1.):
        probs = self.softmax(scores, T=T)
        template_id = np.random.multinomial(1, probs).argmax()
        template = templates.iloc[template_id]
        tfidf_score = tfidf_scores[template_id]
        return template#, tfidf_score

    def softmax(self, scores, T=1.):
        exp_scores = np.exp(scores / T)
        return exp_scores / np.sum(exp_scores)




class Templates(object):
    """Data structure for templates.
    """
    def __init__(self, templates=[], finalized=False):
        self.templates = templates
        self.template_id = len(templates)
        self.finalized = finalized

    @classmethod
    def from_pickle(cls, path):
        templates = read_pickle(path)
        return cls(templates=templates, finalized=True)

    def add_template(self, utterance, dialogue_state):
        raise NotImplementedError

    def finalize(self):
        self.templates = pd.DataFrame(self.templates)
        self.score_templates()
        self.finalized = True

    def save(self, output):
        assert self.finalized
        write_pickle(self.templates, output)

    def score_templates(self):
        sequences = [s.split() for s in self.templates.template.values]
        vocab = build_vocabulary(1, *sequences)
        counter = count_ngrams(3, vocab, sequences, pad_left=True, pad_right=False)
        model = MLENgramModel(counter)
        scores = [-1.*model.entropy(s)*len(s) for s in sequences]
        if not 'logp' in self.templates.columns:
            self.templates.insert(0, 'logp', 0)
        self.templates['logp'] = scores

if __name__ == "__main__":

    g = Generator(persuader=1)
    # g.retrieve('are you going to donate as well', 2, 5)
