#torch version 0.4.1
import torch
import random
import numpy as np
import sys
sys.path.append('/home/wyshi/simulator')
from sequicity.config import global_config as cfg
from sequicity.reader import CamRest676Reader, get_glove_matrix
from sequicity.reader import KvretReader
from sequicity.tsd_net import TSD, cuda_, nan
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
from sequicity.reader import pad_sequences
import argparse, time

from sequicity.metric import CamRestEvaluator, KvretEvaluator
import logging
import pdb
from nltk.tokenize import word_tokenize

class Model:
    def __init__(self, dataset, discrete_act=False):
        reader_dict = {
            'camrest': CamRest676Reader,
            'kvret': KvretReader,
        }
        model_dict = {
            'TSD':TSD
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'kvret': KvretEvaluator,
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                               hidden_size=cfg.hidden_size,
                               vocab_size=cfg.vocab_size,
                               layer_num=cfg.layer_num,
                               dropout_rate=cfg.dropout_rate,
                               z_length=cfg.z_length,
                               max_ts=cfg.max_ts,
                               # action_size=,
                               discrete_act=discrete_act,
                               beam_search=cfg.beam_search,
                               beam_size=cfg.beam_size,
                               eos_token_idx=self.reader.vocab.encode('EOS_M'),
                               vocab=self.reader.vocab,
                               teacher_force=cfg.teacher_force,
                               degree_size=cfg.degree_size,
                               reader=self.reader)
        self.EV = evaluator_dict[dataset] # evaluator class
        if cfg.cuda: self.m = self.m.cuda()
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))

        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        z_input = cuda_(Variable(torch.from_numpy(z_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))

        kw_ret['z_input_np'] = z_input_np

        return u_input, u_input_np, z_input, m_input, m_input_np,u_len, m_len,  \
               degree_input, kw_ret

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)

                    loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                        m_input=m_input,
                                                                        degree_input=degree_input,
                                                                        u_input_np=u_input_np,
                                                                        m_input_np=m_input_np,
                                                                        turn_states=turn_states,
                                                                        u_len=u_len, m_len=m_len, mode='train', **kw_ret)
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()
                    # pdb.set_trace()
                    sup_loss += loss.data.cpu().numpy()
                    sup_cnt += 1
                    logging.debug(
                        'loss:{} pr_loss:{} m_loss:{} grad:{}'.format(loss.data,
                                                                       pr_loss.data,
                                                                       m_loss.data,
                                                                       grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time()-sw))
            valid_loss = valid_sup_loss + valid_unsup_loss
            self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                  weight_decay=5e-5)
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))
                
    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch, prev_z)
                m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                                   m_input=m_input,
                                                   degree_input=degree_input, u_input_np=u_input_np,
                                                   m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                                   dial_id=turn_batch['dial_id'], **kw_ret)
                self.reader.wrap_result(turn_batch, m_idx, z_idx, prev_z=prev_z)
                prev_z = z_idx
        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        self.m.train()
        return res

    def interactive(self):

        turn_batch = {
            'dial_id': [0],
            'turn_num': [0],
            'user': [[0]],
            'response': [[0]],
            'bspan': [[0]],
            'u_len': [0],
            'm_len': [0],
            'degree': [[0, 0, 0, 0, 0]],
            'supervised': [True]
        }

        mode = 'test'
        turn_states = {}
        prev_z = None
        prev_m = []
        turn_num = 0
        # print('starting the conversation, what can I help you?')
        # print('I am looking for a nice restaurant in the center of town.')
        while True:
            usr_utt = input('User: ')
            usr_utt_tokenized = word_tokenize(usr_utt) + ['EOS_U']
            usr_utt_encoded = self.reader.vocab.sentence_encode(usr_utt_tokenized)
            # z_input = ['EOS_Z1', 'EOS_Z3', 'EOS_Z4', 'EOS_Z5', 'EOS_Z2']
            # m_input = []

            if turn_batch['turn_num'][0] > 10 or usr_utt == 'close':
                break;

            if turn_batch['turn_num'] == [0]:
                turn_batch['user'] = [usr_utt_encoded]
            else:
                turn_batch['user'] = [self.reader.vocab.sentence_encode(word_tokenize(prev_m)) + \
                                      [self.reader.vocab.encode('EOS_M')] + \
                                      usr_utt_encoded]
            turn_batch['u_len'] = [len(i) for i in turn_batch['user']]
            turn_batch['m_len'] = [len(i) for i in turn_batch['response']]

            # pdb.set_trace()

            u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
            m_len, degree_input, kw_ret \
                = self._convert_batch(turn_batch, prev_z)

            # print(self.reader.vocab.sentence_decode(turn_batch['user'][0]))
            # print(self.reader.vocab.sentence_decode([i[0] for i in u_input.data.tolist()]))
            # pdb.set_trace()
            m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                               m_input=m_input,
                                               degree_input=degree_input, u_input_np=u_input_np,
                                               m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                               dial_id=turn_batch['dial_id'], **kw_ret)

            print('Sys: ' + self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M'))
            print('Slots: ' + self.reader.vocab.sentence_decode(z_idx[0], eos='EOS_Z2'))

            print('-------------------------------------------------------\n')

            prev_z = z_idx
            prev_m = self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')

            turn_num += 1
            turn_batch['turn_num'] = [turn_num]
            turn_batch['bspan'] = prev_z

    def rl_interactive_single_turn(self, usr_utt, turn_num, prev_m=None, prev_z=None, turn_states={}, np_state=None,
                                   true_z_input=None, rl_test=False):

        turn_batch = {
                        'dial_id': [0],
                        'turn_num': [turn_num],
                        'user': [[0]],
                        'response': [[0]],
                        'bspan': [[0]],
                        'u_len': [0],
                        'm_len': [0],
                        'degree': [[0,0,0,0,0]],
                        'supervised': [True]
        }

        mode = 'test'
        # turn_states = {}
        if turn_num == 0:
            assert turn_states == {}
            assert prev_z is None
            prev_m = []
        else:
            assert prev_m
            assert prev_z is not None

        if not rl_test:
            turn_batch['bspan'] = [self.reader.vocab.sentence_encode(word_tokenize(true_z_input))]
        # else:
        #     # if rl_test or interactive_test, bspan will not be provided
        #     turn_batch['bspan'] = [prev_z]
        # turn_num = 0
        # print('starting the conversation, what can I help you?')
        # print('I am looking for a nice restaurant in the center of town.')
        while True:
            usr_utt = usr_utt.lower()
            usr_utt_tokenized = word_tokenize(usr_utt) + ['EOS_U']
            usr_utt_encoded   = self.reader.vocab.sentence_encode(usr_utt_tokenized)
            # z_input = ['EOS_Z1', 'EOS_Z3', 'EOS_Z4', 'EOS_Z5', 'EOS_Z2']
            # m_input = []

            if turn_batch['turn_num'][0] > 10 or usr_utt == 'close':
                break;


            if turn_batch['turn_num'] == [0]:
                turn_batch['user'] = [usr_utt_encoded]
            else:
                turn_batch['user'] = [self.reader.vocab.sentence_encode(word_tokenize(prev_m.lower())) + \
                                      [self.reader.vocab.encode('EOS_M')] + \
                                      usr_utt_encoded]
            turn_batch['u_len'] = [len(i) for i in turn_batch['user']]
            turn_batch['m_len'] = [len(i) for i in turn_batch['response']]

            # import pdb
            # pdb.set_trace()
            u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
            m_len, degree_input, kw_ret \
                = self._convert_batch(turn_batch, prev_z)

            # import pdb
            # pdb.set_trace()
            if rl_test:
                z_input = None
            else:
                assert z_input is not None
            results_from_model = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                               m_input=m_input,
                                               degree_input=degree_input, u_input_np=u_input_np,
                                               m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                               dial_id=turn_batch['dial_id'], np_state=np_state, **kw_ret)

            if self.m.discrete_act:
                m_idx, z_idx, turn_states, pz_proba = results_from_model
            else:
                m_idx, z_idx, turn_states, pz_proba, mt_proba = results_from_model

            # print('z_idx', z_idx)
            if self.m.discrete_act:
                # print('Sys act logits: ', m_idx)
                pass
            else:
                print('Sys: ' + self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M'))
            print("*"*20)
            print('Slots: ' + self.reader.vocab.sentence_decode(z_idx[0], eos='EOS_Z2'))
            print('*'*20)

            if rl_test:
                prev_z = z_idx
            else:
                prev_z = [[i[0] for i in z_input]]
            # prev_m = self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')
            # print("z_input in model", z_input, type(z_input))
            # print("z_idx in model", z_idx, type(z_idx))
            # import pdb
            # pdb.set_trace()
            turn_num += 1
            turn_batch['turn_num'] = [turn_num]
            # turn_batch['bspan'] = prev_z
            if self.m.discrete_act:
                return turn_num, m_idx, prev_z, turn_states, pz_proba, z_input
            else:
                return turn_num, m_idx, prev_z, turn_states, pz_proba, z_input, mt_proba

    def fill_sentence(self, m_idx, z_idx):

        sent = self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M').split()
        slots = [self.reader.vocab.decode(z) for z in z_idx[0]]
        constraints = slots[:slots.index('EOS_Z1')]
        db_results = self.reader.db_search(constraints)

        filled_sent = []
        filled_slot = {}
        import random
        if db_results:
            rand_result = random.choice(db_results)
            for idx, word in enumerate(sent):
                if '_SLOT' in word:
                    filled_sent.append(rand_result[word.split('_')[0]])
                    filled_slot[word.split('_')[0]] = rand_result[word.split('_')[0]]
                else:
                    filled_sent.append(word)

        # filled_sent = ' '.join(sent)
        return " ".join(sent), ' '.join(filled_sent), filled_slot

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch)

                loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                    m_input=m_input,
                                                                    turn_states=turn_states,
                                                                    degree_input=degree_input,
                                                                    u_input_np=u_input_np, m_input_np=m_input_np,
                                                                    u_len=u_len, m_len=m_len, mode='train',**kw_ret)
                sup_loss += loss.data
                sup_cnt += 1
                logging.debug(
                    'loss:{} pr_loss:{} m_loss:{}'.format(loss.data, pr_loss.data, m_loss.data))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        print('result preview...')
        # self.eval()
        return sup_loss, unsup_loss

    def reinforce_tune(self):
        lr = cfg.lr
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()))
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        for epoch in range(self.base_epoch + cfg.rl_epoch_num + 1):
            mode = 'rl'
            if epoch <= self.base_epoch:
                continue
            epoch_loss, cnt = 0,0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = self.optim #Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=0)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)
                    loss_rl = self.m(u_input=u_input, z_input=z_input,
                                m_input=m_input,
                                degree_input=degree_input,
                                u_input_np=u_input_np,
                                m_input_np=m_input_np,
                                turn_states=turn_states,
                                dial_id=turn_batch['dial_id'],
                                u_len=u_len, m_len=m_len, mode=mode, **kw_ret)

                    if loss_rl is not None:
                        loss = loss_rl #+ loss_mle * 0.1
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 2.0)
                        optim.step()
                        epoch_loss += loss.data.cpu().numpy()[0]
                        cnt += 1
                        logging.debug('{} loss {}, grad:{}'.format(mode,loss.data[0],grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = epoch_loss / (cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss

            #self.save_model(epoch)

            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def save_model(self, epoch, path=None, critical=False):
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')


        print("**"*20)
        print('loaded!!!!!!!!!!!')
        print("**"*20)

        cur_model_state = self.m.state_dict()
        pretrained_dict = all_state['lstd']

        cur_model_state.update(pretrained_dict)

        self.m.load_state_dict(cur_model_state)
        self.m.eval()
        # pdb.set_trace()


        # self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        # pdb.set_trace()
        logging.info('finish loading glove model')

        self.m.u_encoder.embedding.weight = torch.nn.Parameter(embedding_arr)
        # self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)

        # pdb.set_trace()

        self.m.z_decoder.emb.weight = torch.nn.Parameter(embedding_arr)
        self.m.m_decoder.emb.weight = torch.nn.Parameter(embedding_arr)

        # self.m.z_decoder.emb.weight.data.copy_(embedding_arr)
        # self.m.m_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):

        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])

        print('total trainable params: %d' % param_cnt)


def load_rl_model(discrete_act, pretrained_dir=None):
    arg_model = 'tsdf-camrest'
    arg_mode = 'interact'

    cfg.init_handler(arg_model)
    cfg.dataset = arg_model.split('-')[-1]

    logging.info(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = arg_mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(arg_model.split('-')[-1], discrete_act=discrete_act)
    m.count_params()

    if pretrained_dir is None:
        m.load_glove_embedding()
    else:
        m.load_model()
        # pdb.set_trace()
    # m.interactive()

    return m

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.init_handler(args.model)
    cfg.dataset = args.model.split('-')[-1]

    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            else:
                v = dtype(v)
            setattr(cfg, k, v)

    logging.info(str(cfg))
    if cfg.cuda:
        torch.cuda.set_device(cfg.cuda_device)
        logging.info('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    m = Model(args.model.split('-')[-1])
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        m.eval()
    elif args.mode == 'rl':
        m.load_model()
        m.reinforce_tune()

    elif args.mode == 'interact':
        m.load_model()
        m.interactive()
    elif args.mode == 'vocab':
        m.load_glove_embedding()


if __name__ == '__main__':
    main()
