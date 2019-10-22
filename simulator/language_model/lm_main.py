import sys, pdb
sys.path.append('/home/wyshi/simulator')
from simulator.language_model.counter import build_vocabulary, count_ngrams
from simulator.language_model.ngram import MLENgramModel

def score_templates(sents):
    sequences = [s.split() for s in sents]
    vocab = build_vocabulary(1, *sequences)
    counter = count_ngrams(3, vocab, sequences, pad_left=True, pad_right=False)
    model = MLENgramModel(counter)
    scores = [-1. * model.entropy(s) * len(s) for s in sequences]

    return scores



sents = 'no , i would like a expensive restaurant .'
sequences = [s.split() for s in sents]
vocab = build_vocabulary(1, *sequences)
counter = count_ngrams(3, vocab, sequences, pad_left=True, pad_right=False)
model = MLENgramModel(counter)
pdb.set_trace()
scores = [-1. * model.entropy(s) * len(s) for s in sequences]