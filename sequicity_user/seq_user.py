import sys, os, re, pdb
# sys.path.append('/home/wyshi/simulator/sequcity_user/')
# sys.path.append('/data/qkun/sequcity_mulitwoz_0.4/')
sys.path.append('/home/wyshi/simulator/')
import logging, random
import torch
import numpy as np
from nltk import word_tokenize
from collections import defaultdict

from sequicity_user.model import Model
from sequicity_user.config import global_config as cfg

import simulator.dialog_config as dialog_config
import simulator.nlg as nlg
from simulator.user import User
from simulator.agent.core import Action, SystemAct


class Seq_User(User):
    def __init__(self, nlg_sample, nlg_template):
        super().__init__(nlg_sample=nlg_sample, nlg_template=nlg_template)
        self._set_initial_state()

        self._set_initial_goal_dic()

        # # # # # # # # # # # # # # # # 
        # # model configure setting # #

        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        cfg.init_handler('tsdf-usr')
        cfg.dataset = 'usr'
        if cfg.cuda:
            torch.cuda.set_device(cfg.cuda_device)
            logging.info('Device: {}'.format(torch.cuda.current_device()))
        self.m = Model('usr')
        self.m.m = self.m.m.to(device)
        self.m.count_params()
        self.m.load_model()
        self.entity = self.m.reader.entity
        # # # # # # # # # # # # # # # # 

        self.state_list = []

        self._set_initial_model_parameters()


    def _set_initial_state(self):
        self.state = {
                    'informed': {k:0 for k in  self.entity_type['informable_slots']},
                    'asked': {k:0 for k in  self.entity_type['requestable_slots']},
                    'asked_answered': {k:0 for k in  self.entity_type['requestable_slots'] + ['name']},
                    'reservation_informed': {k:0 for k in  self.entity_type['reservation_slots']},
                    'results': [],
                    'no_match_presented': 0,
                    'asked_anything_else': 0,
                    'no_other_presented': 0,
                    'match_presented': 0,
                    'book_fail': 0,

                    'usr_act_sequence': [],
                    'sys_act_sequence': [],

                    'inform': {k:None for k in  self.entity_type['informable_slots']},
                    'book': {k:None for k in  self.entity_type['reservation_slots']}
                    }
        self.check_constrain = []#dialog_config.CONSTRAINT_CHECK_NOTYET
        self.check_info = dialog_config.INFO_CHECK_NOTYET
        self.check_reservation = []#dialog_config.RESERVATION_CHECK_NOTYET
        self.dialog_status = dialog_config.NO_OUTCOME_YET

    def _set_initial_goal_dic(self):
        # # goal transfer into list
        self.goal_dic = defaultdict(list)
        for key in ['cur_info', 'info_second_choice', 'cur_book', 'book_second_choice']:
            if key in self.goal:
                for slot_name in self.goal[key]:
                    self.goal_dic[slot_name] += [self.goal[key][slot_name]]
        if 'reqt' in self.goal:
            for slot_name in self.goal['reqt']:
                self.goal_dic[slot_name] = [slot_name]

        self.goal_list = list(self.goal['cur_info'].keys())
        if 'info_second_choice' in self.goal:
            self.goal_list += list(self.goal['info_second_choice'].keys())
        if 'reqt' in self.goal:
            self.goal_list += list(self.goal['reqt'])
        if 'cur_book' in self.goal:
            self.goal_list += list(self.goal['cur_book'].keys())
        if 'book_second_choice' in self.goal:
            self.goal_list += list(self.goal['book_second_choice'].keys())

    def _set_initial_model_parameters(self):
        self.turn_batch = {
                'dial_id': [0],
                'turn_num': [0],
                'user': [[0]],
                'response': [[0]],
                'bspan': [[0]],
                'u_len': [0],
                'm_len': [0],
                'degree': [[1]],
                'supervised': [True],
                'goal': [self.m.reader.vocab.sentence_encode(word_tokenize(' '.join(self.goal_list)) + ['EOS_Z0'])]
        }
        self.prev_z = None
        self.prev_m = None


    def respond(self, sys_act, prev_sys=None):
        mode = 'test'
        turn_states = {}
        turn_num = self.turn_batch['turn_num'][0]

        if turn_num != 0:
            self.update_states_from_sys(sys_act)

        if prev_sys is None:
            prev_sys = 'Hello! What can I help you?'.lower()
        else:
            prev_sys = prev_sys.lower()

        # # format input
        utt_tokenized = word_tokenize(prev_sys) + ['EOS_U']
        utt_encoded   = self.m.reader.vocab.sentence_encode(utt_tokenized)

        if self.turn_batch['turn_num'] == [0]:
            self.turn_batch['user'] = [utt_encoded]
        else:
            self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_m)) + \
                                 [self.m.reader.vocab.encode('EOS_M')] + \
                                 utt_encoded]

        self.turn_batch['u_len'] = [len(i) for i in self.turn_batch['user']]
        self.turn_batch['m_len'] = [len(i) for i in self.turn_batch['response']]

        u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
            m_len, degree_input, kw_ret \
                = self.m._convert_batch(self.turn_batch, self.prev_z)

        # pdb.set_trace()
        # # execute tsd-net
        m_idx, z_idx, turn_states = self.m.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                           m_input=m_input,
                                           degree_input=degree_input, u_input_np=u_input_np,
                                           m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                           dial_id=self.turn_batch['dial_id'], **kw_ret)

        cur_usr = self.m.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')
        filled_usr, slot_dic = self.fill_sentence(cur_usr)

        if turn_num != 0:
            # pdb.set_trace()
            self.success_or_not(self.prev_m, prev_sys, filled_usr, sys_act)
        self.update_states_from_user(filled_usr)

        # ##############################
        # pdb.set_trace()
        # ################################



        self.prev_z = z_idx
        self.prev_m = filled_usr
        turn_num += 1
        self.turn_batch['turn_num'] = [turn_num]
        # self.turn_batch['bspan'] = self.prev_z

        return None, self.prev_m


    def interact(self):
        mode = 'test'
        turn_states = {}
        turn_num = self.turn_batch['turn_num'][0]
        # utterance = input('User:',).lower()
        utterance = 'Hello! What can I help you?'.lower()
        print('Sys: ' + utterance)
        while True:

            if self.turn_batch['turn_num'][0] > 10 or utterance == 'close':
                break;

            # # format input
            utt_tokenized = word_tokenize(utterance) + ['EOS_U']
            utt_encoded   = self.m.reader.vocab.sentence_encode(utt_tokenized)

            if self.turn_batch['turn_num'] == [0]:
                self.turn_batch['user'] = [utt_encoded]
            else:
                self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_m)) + \
                                     [self.m.reader.vocab.encode('EOS_M')] + \
                                     utt_encoded]

            self.turn_batch['u_len'] = [len(i) for i in self.turn_batch['user']]
            self.turn_batch['m_len'] = [len(i) for i in self.turn_batch['response']]

            u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self.m._convert_batch(self.turn_batch, self.prev_z)

            # # execute tsd-net
            m_idx, z_idx, turn_states = self.m.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                               m_input=m_input,
                                               degree_input=degree_input, u_input_np=u_input_np,
                                               m_input_np=m_input_np, m_len=m_len, turn_states=turn_states,
                                               dial_id=self.turn_batch['dial_id'], **kw_ret)
            
            sent = self.m.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M')
            # print('Usr Simu: ' + sent)

            filled_sent = self.fill_sentence(sent)
            print('Usr Simu: ' + filled_sent)
            # print('Slots: ' + self.m.reader.vocab.sentence_decode(z_idx[0], eos='EOS_Z2'))
            # pdb.set_trace()
            print('Goal:' + ' '.join(self.goal_list))
            print('-------------------------------------------------------\n')
            pdb.set_trace()

            self.prev_z = z_idx
            self.prev_m = filled_sent
            turn_num += 1
            self.turn_batch['turn_num'] = [turn_num]
            # self.turn_batch['bspan'] = self.prev_z


            utterance = input('Sys:',).lower()

    def fill_sentence(self, slot_sent):
        sent = []
        slot_dic = {}
        for word in word_tokenize(slot_sent):
            if '_SLOT' in word:
                slot_name = word.split('_')[0]
                if slot_name not in self.goal_dic:
                    slot_val = random.choice(self.entity['informable'][slot_name])
                    self.goal_dic[slot_name] = [slot_val]
                    # pdb.set_trace()
                else:

                    if len(self.goal_dic[slot_name]) > 1:
                        slot_val = self.goal_dic[slot_name].pop(0)
                    else:
                        slot_val = self.goal_dic[slot_name][0]
                slot_dic[slot_name] = slot_val
                sent.append(slot_val)
            else:
                sent.append(word)               
        return ' '.join(sent), slot_dic

    def success_or_not(self, prev_usr, prev_sys, cur_usr, sys_act):

        # # judge whether stop
        stop_flag = 0
        non_stop_pat = re.compile('number|phone|post|address|name|information|value_|restaurant_')
        
        if 'bye' in cur_usr and \
           '?' not in cur_usr:
            stop_flag = 1
        elif 'thank' in cur_usr and \
             '[' not in cur_usr and \
             '?' not in cur_usr:
            stop_flag = 1
        elif re.match('.*have a (good|nice|lovely).*', cur_usr) and \
             '?' not in cur_usr:
            stop_flag = 1
        elif re.match('.*(that is|thats|that s|that will be) all.*', cur_usr):
            stop_flag = 1
        elif not re.findall(non_stop_pat, cur_usr):
            if 'all set' in cur_usr:
                stop_flag = 1
            elif 'i am all i need' in cur_usr:
                stop_flag = 1
            elif 'that s it' in cur_usr:
                stop_flag = 1

        if self.turn_batch['turn_num'][0] > dialog_config.MAX_TURN:
            stop_flag = 1

        if sys_act.act == SystemAct.NOMATCH_RESULT and 'info_second_choice' not in self.goal:
            stop_flag = 1

        # # system ending

        if sys_act.act == SystemAct.GOODBYE:
            self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        # # ask info
        elif re.findall(r'(?<!reference) number|(?<!reservation) number|phone|post *code| address| name|information', prev_usr):
            if sys_act.act == SystemAct.PROVIDE_INFO:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            
        # # # reservation
        # prev_usr_slot = self.m.reader.delex_sent(prev_usr)
        elif re.search(r'value_time|value_day|value_people', self.m.reader.delex_sent(prev_usr)) is not None or \
           re.search(r'reference number|reservation number', prev_usr) is not None:
            # # reference number
            if sys_act.act == SystemAct.ASK_RESERVATION_INFO:
                tmp_flag = 1
                for slot_name in ['time','day','people']:
                    if slot_name in prev_sys and self.state['book'][slot_name] is not None:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

            elif sys_act.act in [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)

            elif sys_act.act == SystemAct.PRESENT_RESULT:
                prev_sys_slot = self.m.reader.delex_sent(prev_sys)
                constraints = [slot[1:-1].split('|')[1] for slot in re.findall(r'\[.*?\]', prev_sys_slot)]
                tmp_flag = 1
                if self.state['inform']['name'] is not None:
                    tmp_flag = 0
                for slot_name in self.state['inform']:
                    if self.state['inform'][slot_name] is not None and self.state['inform'][slot_name] not in constraints:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)


            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        elif sys_act.act in [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
            self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        elif sys_act.act == SystemAct.ASK_RESERVATION_INFO:
            if 'book' in prev_usr or 'reserv' in prev_usr:
                 self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        # # # inform type

        elif sys_act.act == SystemAct.NOMATCH_RESULT:
            cur_info = {slot_name:slot_val for slot_name, slot_val in self.state['inform'].items() if slot_val is not None}
            match_list = self.query_in_DB(cur_info)
            if not match_list:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)


        elif sys_act.act == SystemAct.NO_OTHER:
            cur_info = {slot_name:slot_val for slot_name, slot_val in self.state['inform'].items() if slot_val is not None}
            match_list = self.query_in_DB(cur_info, skip=self.state['results'])
            if not match_list:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        elif re.search(r'value_area|value_food|value_pricerange', self.m.reader.delex_sent(prev_usr)) is not None:
            if sys_act.act == SystemAct.PRESENT_RESULT:
                prev_sys_slot = self.m.reader.delex_sent(prev_sys)
                constraints = [slot[1:-1].split('|')[1] for slot in re.findall(r'\[.*?\]', prev_sys_slot)]
                tmp_flag = 1
                for slot_name in self.state['inform']:
                    if self.state['inform'][slot_name] is not None and self.state['inform'][slot_name] not in constraints:
                        tmp_flag = 0
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

            elif sys_act.act == SystemAct.ASK_TYPE:
                tmp_flag = 1
                for slot_name in ['area','food','pricerange']:
                    if slot_name in prev_sys and self.state['inform'][slot_name] is not None:
                        tmp_flag = 0
                
                if tmp_flag:
                    self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
                else:
                    self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)              

        elif re.search(r'restaurant_name', self.m.reader.delex_sent(prev_usr)) is not None:
            if sys_act.act == SystemAct.NOMATCH_RESULT or sys_act.act == SystemAct.PRESENT_RESULT:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

        elif sys_act.act == SystemAct.ASK_TYPE:
            if self.state['inform']['name'] is not None and \
               (self.state['inform']['area'] is None or \
                self.state['inform']['food'] is None or \
                self.state['inform']['pricerange'] is None):

                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)

            


        else:
            self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)

        self.dialog_status = self.state_list[-1]

        # print('**********' , self.dialog_status , '************')
        # pdb.set_trace()


        if stop_flag:
            if dialog_config.TURN_FAIL_FOR_SL not in self.state_list:
                self.dialog_status = dialog_config.SUCCESS_DIALOG
            else:
                self.dialog_status = dialog_config.FAILED_DIALOG


    def update_states_from_user(self, cur_usr):
      cur_usr_slot = self.m.reader.delex_sent(cur_usr)
      for slot in re.findall(r'\[.*?\]', cur_usr_slot):
          [slot_name, slot_val] = slot[1:-1].split('|')
          slot_name = slot_name.split('_')[1]
          if slot_name in self.state['inform']:
              self.state['inform'][slot_name] = slot_val
          elif slot_name in self.state['book']:
              self.state['book'][slot_name] = slot_val

    def update_states_from_sys(self, sys_act):
        if sys_act.act == SystemAct.PRESENT_RESULT:
            self.state['results'].append(sys_act.parameters)

    def reset(self):
        super().reset()
        self._set_initial_state()
        self._set_initial_goal_dic()
        self._set_initial_model_parameters()
        self.state_list = []

def main():
    user = Seq_User()
    # user.respond()

    user.interact()

if __name__ == "__main__":
    main()
