import sys, os, re, pdb
# sys.path.append('/home/wyshi/simulator/sequcity_user/')
# sys.path.append('/data/qkun/sequcity_mulitwoz_0.4/')
sys.path.append('/home/wyshi/simulator/')
import logging, random
import torch
import numpy as np
import random
from nltk import word_tokenize
from collections import defaultdict


from sequicity_user.seq_user import Seq_User
from sequicity_user.model import Model
from sequicity_user.config import global_config as cfg

import simulator.dialog_config as dialog_config
import simulator.nlg as nlg
from simulator.user import User
from simulator.agent.core import Action, SystemAct


class Seq_User_Act(Seq_User):
    def __init__(self, nlg_sample, nlg_template):
        super().__init__(nlg_sample=nlg_sample, nlg_template=nlg_template)
        self._set_initial_state()

        self._set_initial_goal_dic()

        # # # # # # # # # # # # # # # # 
        # # model configure setting # #
        cfg.init_handler('tsdf-usr_act')
        cfg.dataset = 'usr_act'
        # logging.info(str(cfg))
        if cfg.cuda:
            torch.cuda.set_device(cfg.cuda_device)
            logging.info('Device: {}'.format(torch.cuda.current_device()))
        self.m = Model('usr_act')
        self.m.count_params()
        self.m.load_model()
        self.entity = self.m.reader.entity
        # # # # # # # # # # # # # # # # 

        self.state_list = []
        self.act = ''
        self.prev_usr = ''

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
                    'book': {k:None for k in  self.entity_type['reservation_slots']},
                    'reqt' : []
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
        self.prev_act = None


    def respond(self, sys_act, prev_sys=None):
        mode = 'test'
        turn_states = {}
        turn_num = self.turn_batch['turn_num'][0]
        act_list = ['inform_type', \
                    'inform_type_change', \
                    'ask_info', \
                    'make_reservation', \
                    'make_reservation_change_time', \
                    'anything_else', \
                    'goodbye']

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
            self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_act)) + \
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

        self.act = act_list[m_idx[0,0,0]]
        if turn_num == 0:
            self.act = 'inform_type'

        # # generating slots
        slot_dict = self.generate_dial_act_slots(sys_act, prev_sys)

        # # generating sentence with templats
        usr_act = Action(self.act, slot_dict)
        # pdb.set_trace()
        # print(usr_act)


        if self.act == 'inform_type' and slot_dict == {} and sys_act.act == SystemAct.ASK_TYPE:
            usr_response_sent = 'i do not care.'
        else:
            if self.nlg_sample:
                assert self.nlg_templates
                assert self.generator
                # usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act, templates=self.nlg_templates, generator=1)
                print('supervised nlg_sample')
                if prev_sys is None:
                    prev_sys = "<start>"

                usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act, templates=self.nlg_templates,
                                                                            generator=self.generator, context=prev_sys,
                                                                            seq2seq=None)
            else:
                # print('')
                if self.seq2seq is None:
                    print("supervised templates")
                    assert self.nlg_template
                    assert not self.nlg_sample
                    assert self.generator is None
                    usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act, turn_num=(len(self.state['usr_act_sequence'])-1),
                                                                                    generator=None,
                                                                                    seq2seq=None)
                else:
                    print(" supervised seq2seq")
                    assert not self.nlg_sample
                    assert not self.nlg_template
                    assert self.seq2seq
                    usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act,
                                                               generator=None,
                                                               seq2seq=self.seq2seq)
                    usr_response_sent = usr_response_sent.replace("<eos>", "")
                usr_response_sent = usr_response_sent.lower()


                # usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act, turn_num=turn_num)

        # # check success of last turn
        if turn_num != 0:
            self.success_or_not(self.prev_usr, prev_sys, usr_response_sent, sys_act)

        # # update states
        self.update_states_from_user(slot_dict)

        self.prev_z = z_idx
        self.prev_act = self.act
        self.prev_usr = usr_response_sent
        turn_num += 1
        self.turn_batch['turn_num'] = [turn_num]
        # self.turn_batch['bspan'] = self.prev_z

        return None, usr_response_sent


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
                self.turn_batch['user'] = [self.m.reader.vocab.sentence_encode(word_tokenize(self.prev_act)) + \
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
            self.prev_act = filled_sent
            turn_num += 1
            self.turn_batch['turn_num'] = [turn_num]
            # self.turn_batch['bspan'] = self.prev_z


            utterance = input('Sys:',).lower()

    def generate_dial_act_slots(self, sys_act, prev_sys):
        slot_dict = {}
        if self.act == 'inform_type':

            avail_slot = []
            if self.turn_batch['turn_num'][0] == 0:
                # avail_slot = self.goal['cur_info'].keys():
                # # pick random number of random slots
                avail_slot = random.sample(self.goal['cur_info'].keys(), k=random.choice(range(1,len(self.goal['cur_info'].keys()) + 1)))
                # slot_dict = self.goal['cur_info']
                for slot in avail_slot:
                    slot_dict[slot] = self.goal['cur_info'][slot]

            elif sys_act.act == SystemAct.ASK_TYPE:
                for slot in ['area', 'food', 'pricerange']:
                    if slot in prev_sys:
                        avail_slot.append(slot)
                        if slot in self.goal['cur_info']:
                            slot_dict[slot] = self.goal['cur_info'][slot]

                if avail_slot == []:
                    # avail_slot = self.goal['cur_info'].keys():
                    slot_dict = self.goal['cur_info']

            else:
                # if sys_act.act == SystemAct.NOMATCH_RESULT:
                #     pdb.set_trace()
                avail_slot = [slot_name for slot_name in self.state['inform'] if self.state['inform'][slot_name] is None]
                if avail_slot:
                    for slot in avail_slot:
                        if slot in self.goal['cur_info']:
                            slot_dict[slot] = self.goal['cur_info'][slot]
                if not slot_dict:
                    if sys_act.act == SystemAct.NOMATCH_RESULT:
                        
                        if 'info_second_choice' in self.goal:
                            self.act = 'inform_type_change'
                            # slot_dict = self.goal['info_second_choice']
                        else:
                            self.act = 'goodbye'
                    elif self.state['results']:
                        if 'reqt' in self.goal:
                            self.act = 'ask_info'
                        else:
                            self.act = 'make_reservation'

        if self.act == 'inform_type_change':
            if 'info_second_choice' not in self.goal:
                # this prediction is bad
                self.act = 'inform_type'
                slot_dict = self.goal['cur_info']
                # pdb.set_trace()
            else:
                # avail_slot = self.goal['info_second_choice'].keys()
                slot_dict = self.goal['info_second_choice']


        if self.act == 'ask_info':
            if self.state['results']:
                if 'reqt' not in self.goal:
                    # this prediction is bad
                    avail_slot = sorted(random.sample(['address','postcode','phone'], k=random.choice(range(1,4))))
                    # pdb.set_trace()
                else:
                    avail_slot = list(set(self.goal['reqt']) - set(self.state['reqt']))


                for slot in avail_slot:
                    slot_dict[slot] = None
                if slot_dict == {}:
                    self.act = 'goodbye'
            else:
                self.act = 'inform_type'
                slot_dict = self.goal['cur_info']

        if self.act == 'make_reservation':
            avail_slot = []
            if self.state['results']:
                if 'cur_book' in self.goal:
                    slot_dict = self.goal['cur_book']

                    # slot_dict = sorted(random.sample(self.goal['cur_book'], k=random.choice(range(1,len(self.goal['cur_book']) + 1))))

                else:
                    if sys_act.act == SystemAct.ASK_RESERVATION_INFO:
                        for slot in ['time', 'day', 'people']:
                            if slot in prev_sys:
                                avail_slot.append(slot)

                    if avail_slot == []:
                        avail_slot = sorted(random.sample(['time', 'day', 'people'], k=random.choice(range(1,4))))

                    for slot in avail_slot:
                        slot_dict[slot] = random.choice(self.entity['informable'][slot])
                    self.goal['cur_book'] = slot_dict
            else:
                self.act = 'inform_type'
                slot_dict = self.goal['cur_info']



        if self.act == 'make_reservation_change_time':
            if 'book_second_choice' in self.goal:
                # avail_slot = self.goal['book_second_choice'].keys()
                slot_dict = self.goal['book_second_choice']
            elif 'make_reservation' in self.goal:
                # this prediction is bad
                # pdb.set_trace()
                self.act = 'make_reservation'
                # avail_slot = self.goal['cur_book'].keys()
                slot_dict = self.goal['cur_book']
            else:
                self.act = 'make_reservation'
                avail_slot = sorted(random.sample(['time', 'day', 'people'], k=random.choice(range(1,4))))
                for slot in avail_slot:
                    slot_dict[slot] = random.choice(self.entity['informable'][slot])
                self.goal['cur_book'] = slot_dict

        else:
            avail_slot = []

        # if slot_dict == {} and self.act == 'inform_type':
        #     pdb.set_trace()
        return slot_dict

    def success_or_not(self, prev_usr, prev_sys, cur_usr, sys_act):

        # # judge whether stop
        stop_flag = 0
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
        # (?# elif re.findall(r'(?<!reference) number|(?<!reservation) number|phone|post *code| address| name|information', prev_usr):)
        elif self.prev_act == 'ask_info':
            if sys_act.act == SystemAct.PROVIDE_INFO:
                self.state_list.append(dialog_config.TURN_SUCCESS_FOR_SL)
            else:
                self.state_list.append(dialog_config.TURN_FAIL_FOR_SL)
            
        # # # reservation
        # prev_usr_slot = self.m.reader.delex_sent(prev_usr)
        elif re.search(r'value_time|value_day|value_people', self.m.reader.delex_sent(prev_usr)) is not None or \
           re.search(r'reference number|reservation number', prev_usr) is not None:
        # elif self.prev_act == 'make_reservation':

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

            elif sys_act.act == SystemAct.PRESENT_RESULT and self.state['results'] == []:
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
        # if self.dialog_status == dialog_config.TURN_FAIL_FOR_SL:
        #     pdb.set_trace()




        if stop_flag:
            if dialog_config.TURN_FAIL_FOR_SL not in self.state_list:
                self.dialog_status = dialog_config.SUCCESS_DIALOG
            else:
                self.dialog_status = dialog_config.FAILED_DIALOG

        # if self.dialog_status == dialog_config.FAILED_DIALOG:
        #     pdb.set_trace()

    def update_states_from_user(self, slot_dic):
        for slot_name in slot_dic:
            slot_val = slot_dic[slot_name]
            if slot_name in self.state['inform']:
                self.state['inform'][slot_name] = slot_val
            elif slot_name in self.state['book']:
                self.state['book'][slot_name] = slot_val
            else:
                self.state['reqt'].append(slot_name)

    def update_states_from_sys(self, sys_act):
        if sys_act.act == SystemAct.PRESENT_RESULT:
            self.state['results'].append(sys_act.parameters)

    def reset(self):
        super().reset()
        self._set_initial_state()
        self._set_initial_goal_dic()
        self._set_initial_model_parameters()
        self.state_list = []
        self.act = ''
        self.prev_usr = ''

def main():
    user = Seq_User_Act()
    # user.respond()

    # user.interact()

if __name__ == "__main__":
    main()
