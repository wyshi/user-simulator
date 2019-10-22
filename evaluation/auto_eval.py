#!/bin/usr/env python3
#
import sys, os, pdb, re
import nltk
sys.path.append('/data/qkun/simulator/data/multiwoz-master/data')
from usr_action_classify import classify_sent
from delex_sent import delex_sent
from collections import defaultdict
from nltk import word_tokenize
sys.path.append('/home/wyshi/simulator/')
from simulator.nlu_model.single_pred_for_qk import usr_act_predictor
import nltk

'''
# # # dialog format # # # 
# dial_id, sys, sys_sent, sys_act
# dial_id, usr, usr_sent, suract
'''

class auto_eval(object):
    def __init__(self, path):

        with open(path, 'r') as dialog:
            self.dial_line = dialog.readlines()

        self.split = ','

        self.usr_utt = self._extract_utt()

        self.delex_usr_utt = self._delex_file()

        self.test_file_path = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant.csv'

        self.test_file = self._extract_test_file(self.test_file_path)

    def _extract_utt(self):
        # # utt : list -> list -> string
        utt = []
        for line in self.dial_line:
            if line.split(',')[1] == 'usr':
                utt.append(word_tokenize(','.join(line.split(',')[2:-1])))
        return utt

    def _delex_file(self):
        delex_usr_utt = []
        for utt in self.usr_utt:
            delex_utt = delex_sent(' '.join(utt)).replace('[', '')
            delex_utt = re.sub(r'\|.*?\]', '', delex_utt)
            delex_usr_utt.append(word_tokenize(delex_utt))
        return delex_usr_utt


    def _extract_test_file(self, path=None):
        test_usr_utt = []
        with open(path, 'r') as test_file:
            for line in test_file.readlines()[1:10000]:
                # pdb.set_trace()
                if line.split(',')[2] == '0':
                    utt = ','.join(line.split(',')[3:]).replace('[','').replace('"','')
                    utt = re.sub(r'\|.*?\]','',utt)
                    test_usr_utt.append(word_tokenize(utt))
        return test_usr_utt


    def _perplexity(self):

        sys.path.append('/home/wyshi/simulator')
        from simulator.language_model.counter import build_vocabulary, count_ngrams
        from simulator.language_model.ngram import LidstoneNgramModel

        sequences = self.delex_usr_utt
        vocab = build_vocabulary(1, *sequences)
        counter = count_ngrams(3, vocab, sequences, pad_left=True, pad_right=False)
        model = LidstoneNgramModel(0.1, counter)

        ppl_per_word = 0
        avg_ppl = 0

        for utt in self.test_file:
            ppl = model.perplexity(' '.join(utt))
            avg_ppl += ppl
            ppl_per_word += ppl / len(utt)
        ppl_per_word /= len(self.test_file)
        avg_ppl /= len(self.test_file)

        return ppl_per_word, avg_ppl

    def mutlti_count(self):
        '''
        this func returns:
        average dialog length(# of turns)
        average utterance length(# of tokens), only for user
        vocab size, only for user
        '''
        dial_num = 0
        token_num = 0.
        turn_num = 0.
        dial_id = ''
        vocab = set()
        for line in self.dial_line:
            if dial_id != line.split(self.split)[0]:
                dial_id = line.split(self.split)[0]
                dial_num += 1
            speaker = line.split(self.split)[1]
            sent = self.split.join(line.split(self.split)[2:-1])
            if speaker == 'usr':
                turn_num += 1
                token_num += len(word_tokenize(sent))
                vocab.update(set(word_tokenize(sent)))
        return turn_num/dial_num, token_num/turn_num, len(vocab)

    def usr_act_dist(self):
        # # this func returns a frequency distribution
        # # of 7 user dialog act
        act_distribution = {'INFORM_TYPE':0,
                            'INFORM_TYPE_CHANGE':0,
                            'ASK_INFO':0,
                            'MAKE_RESERVATION':0,
                            'MAKE_RESERVATION_CHANGE_TIME':0,
                            'ANYTHING_ELSE':0,
                            'GOODBYE':0}

        predictor = usr_act_predictor()
        for utt in self.usr_utt:
            # pdb.set_trace()
            act = predictor.predict(' '.join(utt)).upper()
            act_distribution[act] += 1

        # for line in self.dial_line:
        #     speaker = line.split(',')[1]
        #     if speaker == 'usr':
        #         sent = ','.join(line.split(self.split)[2:-1])
        #         act = line.split(self.split)[-1].replace('\n','').upper()

        #         act_distribution[act] += 1
        return act_distribution



def main():

    # act_tmp = open('/home/wyshi/simulator/evaluation/act_tmp.txt', 'w+')
    for file_name in ['rule_template','rule_sample','rule_generation', \
                      'seq_template', 'seq_sample', 'seq_generation']:
        corpus_path = '/home/wyshi/simulator/evaluation/dial_log/' + file_name + '.csv'
    #     # print('--'*20)
    #     # print(corpus_path)
    # corpus_path = '/home/wyshi/simulator/evaluation/dial_log/seq_generation.csv'
        Eval = auto_eval(corpus_path)

        dial_len, sent_len, vocab_size = Eval.mutlti_count()
        print('average dialog length: %0.3f turns \naverage sentence length: %0.3f words\nvocabulary size: %d\n' \
                %(dial_len, sent_len, vocab_size))

    #     # ppl_per_word, perplexity = Eval._perplexity()
    #     # print('average perplexity per word: %0.3f\naverage perplexity: %0.3f\n' %(ppl_per_word, perplexity))

    #     usr_act_dist = Eval.usr_act_dist()
    #     print(usr_act_dist, '\n')
    #     # pdb.set_trace()
    #     act_tmp.write(' '.join([str(n) for n in usr_act_dist.values()]) + '\n')
    # act_tmp.close()
    # path = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant_non_delex_no_newline.csv'
    # Eval = auto_eval(path)
    # usr_act_dist = Eval.usr_act_dist()
    # print(usr_act_dist, '\n')

if __name__ == '__main__':
    main()





