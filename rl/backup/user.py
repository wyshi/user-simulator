from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle as pkl
from numpy import random
import json
from collections import Counter
from simulator.nlg import UserNlg
from simulator.agent.core import Action, SystemAct, UserAct
import os
from simulator import dialog_config
from copy import deepcopy
cur_dir = os.getcwd()

# area, food, pricerange, phone, address, postcode, name
# with open(os.path.join(cur_dir, "data/cambridge_data/state_by_slots_no_dontcare_improved"), "rb") as fh:
#     slots_by_state = pkl.load(fh)

data_dir = "data/multiwoz-master/data/multi-woz/"
with open("data/multiwoz-master/data/multi-woz/restaurant_db.json", "rb") as fh:
    DB = json.load(fh)



actionID_to_template = {
    0: "thank you for using our system, goodbye .", # arrival
    1: "do you have a [slot_food] preference ?", # departure
    2: "do you have a [slot_area] preference ?", # time
    3: "do you have a [slot_pricerange] preference ?",
    4: "I am sorry, but there are no restaurants matching your request. Is there anything else I can help you with?",#,
    5: "[value_name] is a good restaurant matching your request. Is there anything else I can help you with?",
    6: "[value_name] is located at [value_address] . its [slot_phone] is [value_phone] and the [slot_postcode] is [slot_postcode] . is there anything else i can help you with ?"
}

STATE_TO_SENTS = [None]*10
STATE_TO_SENTS[0] = "thank you, that is all i want, bye"
STATE_TO_SENTS[1] = "anything else"
STATE_TO_SENTS[2] = "restaurant slot_address and slot_phone"
STATE_TO_SENTS[3] = "restaurant slot_address and slot_phone"
STATE_TO_SENTS[4] = "food type/price type"
STATE_TO_SENTS[5] = "thank you good bye"
STATE_TO_SENTS[6] = "i am looking for a restaurant, food, price"
STATE_TO_SENTS[7] = "i am looking for a restaurant, food, price, /sorry, no match"
STATE_TO_SENTS[8] = "thank you bye"
STATE_TO_SENTS[9] = "i am looking for a restaurant, food, price"


# EeNTITY_POOL = {'food': ['dont_care', 'chinese', 'american'],
#                'area': ['dont_care', 'west', 'east'],
#                'pricerange': ['dont_care', 'moderate', 'expensive']}
import pandas as pd
# DB = pd.read_csv(os.path.join(cur_dir, "data/NNDIAL-master/db/CamRest.csv"))
with open(os.path.join(cur_dir, "data/NNDIAL-master/db/ENTITY_POOL_no_dontcare.pkl"), "rb") as fh:
    ENTITY_POOL = pkl.load(fh)
INITIAL_STATE = random.choice([6, 7, 9], size=1)[0]


class Goal(object):
    def __init__(self):
        with open(data_dir + "detailed_goals.pkl", "rb") as fh:
            self.goal_pool = pkl.load(fh)
        self.entity_type = {
            "informable_slots": dialog_config.informable_slots,
            "requestable_slots": dialog_config.requestable_slots,
            "reservation_slots": dialog_config.reservation_slots
        }
        self.DB = DB

    def _intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))

    def query_in_DB(self, cur_info):
        match_list = []
        for restaurant in self.DB:
            match = True
            for entity, value in cur_info.items():
                # assert type(value) == type(value[0])
                # assert isinstance(value, list)
                if restaurant[entity] != value:
                    match = False
            if match:
                match_list.append(restaurant)
        return match_list

    def sample(self, examples, n=1, p=None):
        if p is None:
            # uniform
            p = [1/len(examples)] * len(examples)

        if n == 1:
            return np.random.choice(examples, p=p)
        else:
            return list(np.random.choice(examples, n, replace=False, p=p))

    def generate_initial_goal(self, goal=None):
        """
        {('book', 'fail_book', 'fail_info', 'info'): 266,
        ('fail_info', 'info', 'reqt'): 852,
        ('book', 'fail_book', 'fail_info', 'hotel', 'info', 'restaurant'): 192}

        #todo, anything else, not represented in the 'details', only in 676woz-data
        {'id': 'WOZ20006.json',
        'message': "Task 07425: You are looking for a cheap restaurant and it should be in the north part of town.
                    ---Don't go for the first venue--- the system offers you, ask if there is anything else. Make sure you get the phone number of the venue.",
        'details': {'info': {'pricerange': 'cheap', 'area': 'north'},
         'reqt': ['phone'],
         'fail_info': {}}}
        :return:
        """
        if goal is None:
            goal = self.sample(self.goal_pool)
        tmp_goal = {'id': goal['id']}
        if goal['details']['fail_info'] == {} or ('name' in goal['details']['info']):
            # if fails, say goodbye, no second option
            for k, v in goal['details']['info'].items():
                if v == 'north indian':
                    print(goal)
            tmp_goal['cur_info'] = {k: v for k, v in goal['details']['info'].items() if v != dialog_config.I_DO_NOT_CARE}
            if ('name' not in tmp_goal['cur_info']) and self.sample([True, False], p=[0.2, 0.8]):
                tmp_goal['anything_else'] = True
        else:
            #inverse of 'book', the thing in 'fail_info' is actually always the first choise, if fails, then we go to 'info'
            first_choice = goal['details']['fail_info']
            second_choice = goal['details']['info']
            for k, v in goal['details']['info'].items():
                if v == 'north indian':
                    print(goal)
            for k, v in goal['details']['fail_info'].items():
                if v == 'north indian':
                    print(goal)

            """
            if len(goal['details']['fail_info']) > len(goal['details']['info']):
                first_choice = goal['details']['fail_info']
                second_choice = goal['details']['info']
            else:
                first_choice = goal['details']['info']
                second_choice = goal['details']['fail_info']
            """
            tmp_goal['info_first_choice'] = {k: v for k, v in first_choice.items() if v != dialog_config.I_DO_NOT_CARE}
            tmp_goal['info_second_choice'] = {k: v for k, v in second_choice.items() if v != first_choice[k] \
                                              and v != dialog_config.I_DO_NOT_CARE}

            tmp_goal['cur_info'] = deepcopy(tmp_goal['info_first_choice'])
            # for k, v in second_choice.items():
            #     tmp_goal['fail_info'][k] = [v]#
            # for k, v in second_choice.items():
            #     if v not in tmp_goal['info'][k]:
            #         tmp_goal['info'][k].append(v)
        if 'reqt' in goal['details']:
            # the second task is request information task
            if len(self._intersection(self.entity_type['requestable_slots'],
                                      goal['details']['reqt'])) == 0:
                tmp_goal['reqt'] = self.entity_type['requestable_slots']
            else:
                tmp_goal['reqt'] = [v for v in goal['details']['reqt'] if v in self.entity_type['requestable_slots']]
            # tmp_goal['reqt'] = [v for v in goal['details']['reqt']]
        else:
            # the second task is to make a reservation
            if goal['details']['fail_book'] == {}:
                # if fails, say goodbye, no second option
                tmp_goal['cur_book'] = {k: v for k, v in goal['details']['book'].items() if k in ['day', 'people', 'time']}
            else:
                # the thing in 'book' is actually always the first choise, if fails, then we go to 'fail_book'
                if len(goal['details']['fail_book']) > len(goal['details']['book']):
                    first_choice = goal['details']['fail_book']
                    second_choice = goal['details']['book']
                else:
                    first_choice = goal['details']['book']
                    second_choice = goal['details']['fail_book']
                tmp_goal['book_first_choice'] = {k: v for k, v in first_choice.items() if k in ['day', 'people', 'time']}
                tmp_goal['book_second_choice'] = {k: v for k, v in second_choice.items() if v != first_choice[k] and \
                                                  (k in ['day', 'people', 'time'])}

                tmp_goal['cur_book'] = deepcopy(tmp_goal['book_first_choice'])

                # tmp_goal['book'] = {k: [v] for k, v in first_choice.items()}
                # tmp_goal['fail_book'] = {k: [v] for k, v in first_choice.items()}
                # for k, v in second_choice.items():
                #     tmp_goal['fail_book'][k] = [v]  #

                # for k, v in second_choice.items():
                #     if v not in tmp_goal['book'][k]:
                #         tmp_goal['book'][k].append(v)

        # if ('info_first_choice' in tmp_goal) or ('info_second_choice' in tmp_goal):
        #     tmp_goal['first_choice_match'] = len(self.query_in_DB(tmp_goal['info_first_choice']))
        #     tmp_goal['second_choice_match'] = len(self.query_in_DB(tmp_goal['info_second_choice']))
        # else:
        tmp_goal['first_choice_match'] = len(self.query_in_DB(tmp_goal['cur_info']))

        # assert len(tmp_goal) == 2 # (info, book) or (info, request)
        self._original_goal = deepcopy(tmp_goal) # a copy of the original goal
        self.goal = deepcopy(tmp_goal)

        """
        goal['total_query'] = random.choice(3, size=1, p=total_query_p)[0] + 1
        for _ in range(goal['total_query']):
            cur_goal = self.generate_one_goal_entity()
            goal['goal_entity'].append(cur_goal)

        goal['match_nums'] = list(random.choice(range(4), size=1))#self.query_in_DB()
        self.goal = goal
        """

        templates = self.generate_templates(self.goal)
        return self.goal, templates

    def generate_templates(self, goal):
        # raise NotImplementedError
        templates = {'start': ['You are looking for a <b>particular restaurant</b>.',
                               'You are looking for a <b>restaurant</b>.'],
                     'restaurant': {'name': 'Its name is called <b></b>.',
                                    'food': 'The restaurant should serve <b></b> food.',
                                    'area': 'The restaurant should be in the <b></b> area.',
                                    'pricerange': 'The restaurant should be in the <b></b> price range.',
                                    'dontcare': 'You <b>don\'t care</b> about the <b></b>.'},
                     'restaurant_2nd': 'If there is no such restaurant, how about one ',# serves <b></b> food'
                     'anything_else': 'Don\'t go for the first restaurant the system offers you, ask if there is anything else.',
                     'reqt': 'Make sure you get the ',
                     'reservation': ['Once you find the <b>restaurant</b> you want to book a table for <b>people</b> at <b>time</b> on <b>day</b>.',
                                     'If the booking fails, how about <b></b>?',
                                     'Make sure you get the <b>reference number</b>'],
                    }
        sents = []
        if 'name' in goal['cur_info']:
            sents.append(templates['start'][0])
        else:
            sents.append(templates['start'][1])

        for entity, value in goal['cur_info'].items():
            if value != 'dontcare':
                tmp_sent = templates['restaurant'][entity].replace("<b></b>", "<b>"+value+"</b>")
            else:
                tmp_sents = templates['restaurant']['dontcare'].replace("<b></b>", entity)
            sents.append(tmp_sent)

        if 'anything_else' in goal:
            sents.append(templates['anything_else'])

        if 'info_second_choice' in goal:
            assert len(goal['info_second_choice']) <= 3
            if len(goal['info_second_choice']) == 3:
                tmp_sent = 'that serves <b>' + goal['info_second_choice']['food'] + '</b> food in the <b>' + goal['info_second_choice']['area'] + \
                    '</b> area and in the <b>' + goal['pricerange'] + '</b> price range?'
            elif len(goal['info_second_choice']) == 2:
                if ('food' in goal['info_second_choice']) and ('area' in goal['info_second_choice']):
                    tmp_sent = 'that serves <b>' + goal['info_second_choice']['food'] + '</b> food in the <b>' + goal['info_second_choice']['area'] + '</b> area?'

                elif ('food' in goal['info_second_choice']) and ('pricerange' in goal['info_second_choice']):
                    tmp_sent = 'that serves <b>' + goal['info_second_choice']['food'] + '</b> food in the <b>' + \
                               goal['info_second_choice']['pricerange'] + '</b> area?'

                elif ('area' in goal['info_second_choice']) and ('pricerange' in goal['info_second_choice']):
                    tmp_sent = 'that\'s in the <b>' + goal['info_second_choice']['area'] + '</b> area and in the <b>' + \
                               goal['info_second_choice']['pricerange'] + '</b> price range?'
            elif len(goal['info_second_choice']) == 1:
                if 'food' in goal['info_second_choice']:
                    tmp_sent = 'that serves <b>' + goal['info_second_choice']['food'] + '</b> food.'
                elif 'area' in goal['info_second_choice']:
                    tmp_sent = 'that\'s in the <b>' + goal['info_second_choice']['area'] + '</b> area.'
                elif 'pricerange' in goal['info_second_choice']:
                    tmp_sent = 'that\'s in the <b>' + goal['info_second_choice']['pricerange'] + '</b> price range.'

            tmp_sent = templates['restaurant_2nd'] + tmp_sent
            sents.append(tmp_sent)

        if 'reqt' in goal:
            reqt = [r if r != 'phone' else 'phone number' for r in goal['reqt']]
            reqt = ["<b>" + r + '</b>' for r in reqt]
            if len(reqt) == 1:
                tmp_sent = templates['reqt'] + reqt[-1] + "."
            else:
                tmp_sent = ", ".join(reqt[:-1])
                tmp_sent = templates['reqt'] + tmp_sent + ", and " + reqt[-1] + "."
            sents.append(tmp_sent)

        if 'cur_book' in goal:
            assert len(goal['cur_book']) == 3
            tmp_sent = templates['reservation'][0]
            for entity, value in goal['cur_book'].items():
                if entity == 'people':
                    tmp_sent = tmp_sent.replace("<b>"+entity+"</b>", "<b>" + str(value) + " people</b>")
                elif entity == 'day':
                    tmp_sent = tmp_sent.replace("<b>"+entity+"</b>", "<b>" + value.capitalize() + "</b>")
                elif entity == 'day':
                    tmp_sent = tmp_sent.replace("<b>" + entity + "</b>", "<b>" + str(value) + "</b>")
            sents.append(tmp_sent)

        if ('book_second_choice' not in goal) and ('cur_book' in goal):
            sents.append(templates['reservation'][2] + ".")

        elif 'book_second_in_choice' in goal:
            tmp_sent = templates['reservation'][1]
            assert len(goal['book_second_choice']) == 1
            for entity, value in goal['book_second_choice'].items():
                assert entity in ['time', 'day']
                tmp_sent = tmp_sent.replace("<b></b>", "<b>" + value + "</b>")
            sents.append(tmp_sent)
            sents.append(templates['reservation'][2] + ".")

        final_sents = " ".join(sents)

        return final_sents

"""
tmp_sents = []
tmp_goals = []
goal_generator = Goal()
for goal in goal_generator.goal_pool:
    tmp_goal, _ = goal_generator.generate_initial_goal(goal)
    tmp_goals.append(tmp_goal)
    tmp_sents.append(goal_generator.generate_templates(tmp_goal))

import pandas as pd
pd.Series(tmp_sents).to_csv(data_dir+"tmp_goals.csv", encoding='utf-8')
"""

#TODO the second query?, "dont_care" value
class User(object):
    ## class DialogState(object):
    def __init__(self, clean_prob=0.95, nlg_sample=False,
                 transition_prob=None, slots_by_state=None):
        with open(data_dir + "detailed_goals.pkl", "rb") as fh:
            self.goal_pool = pkl.load(fh)
        self.goal_generator = Goal()
        self.DB = DB
        self.nlg_sample = nlg_sample
        if self.nlg_sample:
            with open("data/multiwoz-master/data/multi-woz/act_to_utt_dict_modified.pkl", "rb") as fh:
                self.nlg_templates = pkl.load(fh)
        else:
            self.nlg_templates = None
        self.entity_type = {
            "informable_slots": dialog_config.informable_slots,
            "requestable_slots": dialog_config.requestable_slots,
            "reservation_slots": dialog_config.reservation_slots
        }

        self.max_turn = dialog_config.MAX_TURN
        self.transition_prob = transition_prob # state transition prob
        self.slots_by_state = slots_by_state   # available sentences to sample from with slots
        self.entity_pool = ENTITY_POOL
        self.first_utt = True
        self.INITIAL_STATE = random.choice([6, 7, 9], size=1)[0]

        self.nlg = UserNlg(domain=None, complexity=None)

        # success or not
        # self.done = False
        # self.success = None

        # self.num_entity = num_entity
        # self.initial_action = None
        # self.id_to_entity = {
        #     0: "food",
        #     1: "area",
        #     2: "pricerange",
        #     3: "address",
        #     4: "postcode",
        #     5: "phone"
        # }
        # self.num_action = num_action

        # set initial goals
        self.system_action = []
        # self.not_in_sample = 0
        # self.total_sample = 0
        self.clean_prob=clean_prob
        # self.is_sample = is_sample

        # a series of latent states
        self.states_series = [self.INITIAL_STATE]
        self.response_history = []

        self.initialize_episode()

    def sample(self, examples, n=1, p=None):
        if p is None:
            # uniform
            p = [1/len(examples)] * len(examples)

        if n == 1:
            return np.random.choice(examples, p=p)
        else:
            return list(np.random.choice(examples, n, replace=False, p=p))

    def _intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))

    def set_environment(self, env, sys):
        self.env = env
        self.sys = sys

    def initialize_episode(self):
        goal, goal_templates = self.goal_generator.generate_initial_goal()
        self.goal = goal
        self._set_initial_state()

    def _set_initial_goal(self, total_query_p=[1, 0, 0.0]):
        """
        {('book', 'fail_book', 'fail_info', 'info'): 266,
        ('fail_info', 'info', 'reqt'): 852,
        ('book', 'fail_book', 'fail_info', 'hotel', 'info', 'restaurant'): 192}

        #todo, anything else, not represented in the 'details', only in 676woz-data
        {'id': 'WOZ20006.json',
        'message': "Task 07425: You are looking for a cheap restaurant and it should be in the north part of town.
                    ---Don't go for the first venue--- the system offers you, ask if there is anything else. Make sure you get the phone number of the venue.",
        'details': {'info': {'pricerange': 'cheap', 'area': 'north'},
         'reqt': ['phone'],
         'fail_info': {}}}
        :return:
        """

        goal = self.sample(self.goal_pool)
        tmp_goal = {'id': goal['id']}
        if goal['details']['fail_info'] == {}:
            # if fails, say goodbye, no second option
            tmp_goal['cur_info'] = {k: v for k, v in goal['details']['info'].items()}
        else:
            #inverse of 'book', the thing in 'fail_info' is actually always the first choise, if fails, then we go to 'info'
            first_choice = goal['details']['fail_info']
            second_choice = goal['details']['info']
            """
            if len(goal['details']['fail_info']) > len(goal['details']['info']):
                first_choice = goal['details']['fail_info']
                second_choice = goal['details']['info']
            else:
                first_choice = goal['details']['info']
                second_choice = goal['details']['fail_info']
            """
            tmp_goal['info_first_choice'] = {k: v for k, v in first_choice.items()}
            tmp_goal['info_second_choice'] = {k: v for k, v in second_choice.items() if v != first_choice[k]}

            tmp_goal['cur_info'] = deepcopy(tmp_goal['info_first_choice'])
            # for k, v in second_choice.items():
            #     tmp_goal['fail_info'][k] = [v]#
            # for k, v in second_choice.items():
            #     if v not in tmp_goal['info'][k]:
            #         tmp_goal['info'][k].append(v)
        if 'reqt' in goal['details']:
            # the second task is request information task
            if len(self._intersection(self.entity_type['requestable_slots'],
                                      goal['details']['reqt'])) == 0:
                tmp_goal['reqt'] = self.entity_type['requestable_slots']
            else:
                tmp_goal['reqt'] = [v for v in goal['details']['reqt'] if v in self.entity_type['requestable_slots']]
        else:
            # the second task is to make a reservation
            if goal['details']['fail_book'] == {}:
                # if fails, say goodbye, no second option
                tmp_goal['cur_book'] = {k: v for k, v in goal['details']['book'].items() if k in ['day', 'people', 'time']}
            else:
                # the thing in 'book' is actually always the first choise, if fails, then we go to 'fail_book'
                if len(goal['details']['fail_book']) > len(goal['details']['book']):
                    first_choice = goal['details']['fail_book']
                    second_choice = goal['details']['book']
                else:
                    first_choice = goal['details']['book']
                    second_choice = goal['details']['fail_book']
                tmp_goal['book_first_choice'] = {k: v for k, v in first_choice.items() if k in ['day', 'people', 'time']}
                tmp_goal['book_second_choice'] = {k: v for k, v in second_choice.items() if v != first_choice[k] and \
                                                  (k in ['day', 'people', 'time'])}

                tmp_goal['cur_book'] = deepcopy(tmp_goal['book_first_choice'])

                # tmp_goal['book'] = {k: [v] for k, v in first_choice.items()}
                # tmp_goal['fail_book'] = {k: [v] for k, v in first_choice.items()}
                # for k, v in second_choice.items():
                #     tmp_goal['fail_book'][k] = [v]  #

                # for k, v in second_choice.items():
                #     if v not in tmp_goal['book'][k]:
                #         tmp_goal['book'][k].append(v)

        # if ('info_first_choice' in tmp_goal) or ('info_second_choice' in tmp_goal):
        #     tmp_goal['first_choice_match'] = len(self.query_in_DB(tmp_goal['info_first_choice']))
        #     tmp_goal['second_choice_match'] = len(self.query_in_DB(tmp_goal['info_second_choice']))
        # else:
        tmp_goal['first_choice_match'] = len(self.query_in_DB(tmp_goal['cur_info']))

        # assert len(tmp_goal) == 2 # (info, book) or (info, request)
        self._original_goal = deepcopy(tmp_goal) # a copy of the original goal
        self.goal = deepcopy(tmp_goal)

        """
        goal['total_query'] = random.choice(3, size=1, p=total_query_p)[0] + 1
        for _ in range(goal['total_query']):
            cur_goal = self.generate_one_goal_entity()
            goal['goal_entity'].append(cur_goal)

        goal['match_nums'] = list(random.choice(range(4), size=1))#self.query_in_DB()
        self.goal = goal
        """

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
            'sys_act_sequence': []
        }
        self.check_constrain = []#dialog_config.CONSTRAINT_CHECK_NOTYET
        self.check_info = dialog_config.INFO_CHECK_NOTYET
        self.check_reservation = []#dialog_config.RESERVATION_CHECK_NOTYET
        self.dialog_status = dialog_config.NO_OUTCOME_YET

    def query_in_DB(self, cur_info):
        match_list = []
        for restaurant in self.DB:
            match = True
            for entity, value in cur_info.items():
                # assert type(value) == type(value[0])
                # assert isinstance(value, list)
                if restaurant[entity] != value:
                    match = False
            if match:
                match_list.append(restaurant)
        return match_list

    def reset(self):
        """
        clean up the memory and come up with a new starting entity
        <d>, <a>, <time>, <uncovered_d>, <uncovered_a>
        self.user_action = self.user.reset()
        :return:
        """
        self.initialize_episode()
        self.first_utt = True

        self.INITIAL_STATE = random.choice([6, 7, 9], size=1)[0]
        self.states_series = [self.INITIAL_STATE] # cleans up the latent states
        # which_entity = random.randint(low=0, high=3, size=1)[0]
        # print "not in sample: ", self.not_in_sample/self.total_sample
        self.system_action = [] # clean up the history
        self.not_in_sample = 0
        self.total_sample = 0
        self.response_history = []
        # response = self.respond(system_utt=system_utt, dialog_state=dialog_state)
        # return response

    def add_noise_to_response(self, res):
        clean = np.random.choice(range(2), p=[1-self.clean_prob, self.clean_prob])
        if clean:
            return res
        else:
            return ">NOISE<"

    def sample_entities(self, provide_info, how_many_to_provide=None, entity_type='informed'):
        """
        return a dictionary
        :param provide_info:
        :param entity_type: 'informed'/'reservation_informed'/'asked'
        :return:
        """
        still_available_entities = [k for k in provide_info if self.state[entity_type][k] < len(provide_info[k])]
        if how_many_to_provide is None:
            how_many_to_provide = self.sample(range(len(still_available_entities))) + 1

        what_to_provide = self.sample(still_available_entities, n=how_many_to_provide)

        # print(what_to_provide)
        if isinstance(what_to_provide, str):
            # print(what_to_provide)
            what_to_provide = [what_to_provide]

        parameter_dict = {slot: provide_info[slot] for slot in what_to_provide}

        return parameter_dict

    def _generate_params(self, usr_act_str, sys_act=None):
        if usr_act_str == UserAct.MAKE_RESERVATION:
            if sys_act is None:
                # the first time the user MAKE_RESERVATION
                all_booking_info = self.goal['cur_book']
                sampled_params = self.sample_entities(all_booking_info, how_many_to_provide=None, entity_type='reservation_informed')

                # update states
                for entity in sampled_params:
                    self.state['reservation_informed'][entity] += 1
                return sampled_params
            else:
                # after the user MAKE_RESERVATION, the system ask for more reservation_info
                assert sys_act.act == SystemAct.ASK_RESERVATION_INFO

                params = {}
                for entity in sys_act.parameters:
                    params[entity] = self.goal['cur_book'][entity]

                # update states
                for entity in sys_act.parameters:
                    self.state['reservation_informed'][entity] += 1

                return params


    def rule_policy(self, sys_act=None):
        """
        rule-based policy, user always talks first
        # task 1. restaurant recommendation
        ASK_TYPE = "ask_type"
        PRESENT_RESULT = "present_result"
        NOMATCH_RESULT = "nomatch_result"
        PROVIDE_INFO = "provide_info"

        # task 2. reservation
        ASK_RESERVATION_INFO = "ask_reservation_info"
        BOOKING_SUCCESS = "booking_success"
        BOOKING_FAIL = "booking_fail"
        #REFERENCE_NUM = "reference_num"
        GOODBYE = "goodbye"

        :return:
        """
        if sys_act is None:
            assert self.first_utt
        else:
            if sys_act.act == SystemAct.GOODBYE:
                self.evaluate_GOOD_BYE(sys_act)
                return Action(UserAct.GOODBYE, None)

        # user always talks first, and the first action is always UserAct.INFORM_TYPE
        if self.first_utt:
            usr_act = self.response_FIRST(sys_act=None)
            self.first_utt = False
        else:
            if sys_act.act == SystemAct.ASK_TYPE:
                usr_act = self.response_ASK_TYPE(sys_act)

            elif sys_act.act == SystemAct.PRESENT_RESULT:
                usr_act = self.response_PRESENT_RESULT(sys_act)

            elif sys_act.act == SystemAct.NOMATCH_RESULT:
                usr_act = self.response_NOMATCH_RESULT(sys_act)

            elif sys_act.act == SystemAct.PROVIDE_INFO:
                usr_act = self.response_PROVIDE_INFO(sys_act)

            elif sys_act.act == SystemAct.NO_OTHER:
                usr_act = self.response_NO_OTHER(sys_act)

            elif sys_act.act == SystemAct.ASK_RESERVATION_INFO:
                usr_act = self.response_ASK_RESERVATION_INFO(sys_act)

            elif sys_act.act == SystemAct.BOOKING_SUCCESS:
                usr_act = self.response_BOOKING_SUCCESS(sys_act)

            elif sys_act.act == SystemAct.BOOKING_FAIL:
                usr_act = self.response_BOOKING_FAIL(sys_act)

            elif sys_act.act == SystemAct.GOODBYE:
            #     usr_act = self.response_GOOD_BYE(sys_act)
                raise ValueError("system action %s" % sys_act.act)

            else:
                raise ValueError("Unknown system action %s" % sys_act.act)

        if usr_act.act == UserAct.GOODBYE:
            #self.episode_over = True
            self.evaluate_GOOD_BYE(sys_act)

        if sys_act is not None:
            self.check_pair(sys_act.act)

        if len(self.state['sys_act_sequence']) >= self.max_turn and usr_act.act != UserAct.GOODBYE:
            print("Maximum dialog length reached!")
            self.dialog_status = dialog_config.FAILED_DIALOG


        return usr_act

    def check_presented_result(self, match):
        """
        checke the presented_result/no_match_result
        :return:
        """
        if match == dialog_config.NO_MATCH:
            query_result = self.query_in_DB(self.goal['cur_info'])
            if len(query_result) == 0:
                return dialog_config.CONSTRAINT_CHECK_SUCCESS
            else:
                print("There is at least one match {}".format(query_result[0]))
                return dialog_config.CONSTRAINT_CHECK_FAILURE
        elif match == dialog_config.NO_OTHER:
            if self.state['usr_act_sequence'][-1] == UserAct.ANYTHING_ELSE and self.state['sys_act_sequence'][-2] == SystemAct.PRESENT_RESULT:
                ###################################
                # can only be the response of ANYTHING_ELSE, and present_result is also the previous response
                # the only correct sequence is sys: present_result -> usr: anything_else -> sys: no_other
                ###################################
                query_result = self.query_in_DB(self.goal['cur_info'])
                if len(query_result) == 1:
                    # indeed there is only one match
                    return dialog_config.CONSTRAINT_CHECK_SUCCESS
                elif len(query_result) == 0:
                    print("There is no match at all for the constrain from the very beginning!")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
                else:
                    print("There are more than one match for the constrain! should present the second result!")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
            else:
                ###################################
                # can only be the response of ANYTHING_ELSE, and present_result already existed before
                # the only correct sequence is sys: present_result -> usr: anything_else -> sys: no_other
                ###################################
                if self.state['usr_act_sequence'][-1] != UserAct.ANYTHING_ELSE:
                    print("FAIL, because the user didn't ask for anything_else")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
                elif self.state['sys_act_sequence'][-2] != SystemAct.PRESENT_RESULT:
                    print("FAIL, because the last sys act is not present_result")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE
        else:
            all_requirements_informed = [(self.state['informed'][entity] > 0) for entity in self.goal['cur_info']]
            all_requirements_informed = np.all(all_requirements_informed)
            if all_requirements_informed:
                for k, v in self.goal['cur_info'].items():
                    if v != match[k] and v != dialog_config.I_DO_NOT_CARE:
                        print("the presented_result doesn't match the requirement!")
                        return dialog_config.CONSTRAINT_CHECK_FAILURE
                return dialog_config.CONSTRAINT_CHECK_SUCCESS
            else:
                # the user hasn't informed all the slots
                tmp_constraint_check = [(self.goal['cur_info'][entity] == match[entity]) for entity, value in self.state['informed'].items() \
                                        if ((value > 0) and (entity in self.goal['cur_info']) and (self.goal['cur_info'][entity] != dialog_config.I_DO_NOT_CARE))]

                if len(tmp_constraint_check) and np.all(tmp_constraint_check):
                    print("Warning, the system hasn't captured all the correct entity but gives the result anyway")
                    return dialog_config.CONSTRAINT_CHECK_SUCCESS
                else:
                    print("Warning, the system hasn't captured all the correct entity but gives the result anyway, and the result is not correct")
                    return dialog_config.CONSTRAINT_CHECK_FAILURE

                return dialog_config.CONSTRAINT_CHECK_FAILURE
                raise ValueError("the user hasn't informed all requirements! but the system presents the result already.")

    def check_provided_info(self, info):
        """
        check the provided_info
        :param info:
        :return:
        """
        if len(self.state['results']) == 0:
            print("FAIL: haven't presented any result yet!")
            return dialog_config.INFO_CHECK_FAILURE
        else:
            if self.state['usr_act_sequence'][-1] in [UserAct.ASK_INFO]:
                # must be the immediate response to ASK_INFO
                restaurant = self.state['results'][-1]
                all_met = [info[entity] == restaurant[entity] for entity in self.goal['reqt'] if entity in restaurant]
                all_met = np.all(all_met)
                if all_met:
                    return dialog_config.INFO_CHECK_SUCCESS
                else:
                    print("FAIL: the info doesn't match the presented restaurant's info!\n{}\n{}".format(restaurant, info))
                    return dialog_config.INFO_CHECK_FAILURE
            else:
                print("Fail: the last user act is {}, not the immediate response of ask_info".format(self.state['usr_act_sequence'][-1]))
                return dialog_config.INFO_CHECK_FAILURE

    def check_reservation_result(self, sys_act_str):
        if self.state['usr_act_sequence'][-1] in [UserAct.MAKE_RESERVATION, UserAct.MAKE_RESERVATION_CHANGE_TIME]:
            # immediate response to MAKE_RESERVATION/MAKE_RESERVATION_CHANGE_TIME
            all_reservation_informed = [(self.state['reservation_informed'][entity] > 0) for entity in self.goal['cur_book']]
            all_reservation_informed = np.all(all_reservation_informed)
            if all_reservation_informed:
                if sys_act_str in [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                    return dialog_config.RESERVATION_CHECK_SUCCESS
                else:
                    return dialog_config.RESERVATION_CHECK_FAILURE
            else:
                if sys_act_str != SystemAct.ASK_RESERVATION_INFO:
                    print("Fail: the user hasn't informed all the reservation info yet, but the system has already made a reservation")
                    return dialog_config.RESERVATION_CHECK_FAILURE
                else:
                    return dialog_config.RESERVATION_CHECK_SUCCESS
        else:
            print("Fail: the last user act is {}, not the immediate previous response of make_reservation/ask_reservation_info".format(self.state['usr_act_sequence'][-1]))
            return dialog_config.RESERVATION_CHECK_FAILURE

    def response_FIRST(self, sys_act):
        # must be inform_type
        # checked
        provide_info = self.goal['cur_info']
        entity_dict = self.sample_entities(provide_info)

        # update state
        for entity in entity_dict:
            self.state['informed'][entity] += 1
        return Action(UserAct.INFORM_TYPE, entity_dict)

    def response_ASK_TYPE(self, sys_act):
        # ask_type --> inform_type
        # checked
        assert sys_act.act == SystemAct.ASK_TYPE
        possible_actions = [UserAct.INFORM_TYPE]
        selected_action = self.sample(possible_actions)

        # update state
        for entity in sys_act.parameters:
            self.state['informed'][entity] += 1

        params = {}
        for entity in sys_act.parameters:
            if entity in self.goal['cur_info']:
                params[entity] = self.goal['cur_info'][entity]
            else:
                params[entity] = dialog_config.I_DO_NOT_CARE
        return Action(selected_action, params)

    def response_PRESENT_RESULT(self, sys_act):
        # present_result --> (ask_info, anything_else, make_reservation)
        # update states
        # checked
        self.state['match_presented'] += 1
        self.state['results'].append(sys_act.parameters)
        self.check_constrain.append(self.check_presented_result(match=sys_act.parameters))

        possible_actions = []

        restaurant_info = sys_act.parameters
        requirement_match = [restaurant_info[entity] == value for entity, value in self.goal['cur_info'].items()]
        requirement_match = np.all(requirement_match)
        if not requirement_match:
            possible_actions.append(UserAct.INFORM_TYPE)
        else:
            # else:# only the presented results match all the requirements in the goal
            if self.state['asked_anything_else'] < dialog_config.AT_MOST_ANYTHING_ELSE:
                if 'name' not in self.goal['cur_info']:# and self.goal['anything_else']:
                    # if ask for a specific restaurant, cannot ask for anything_else
                    possible_actions = [UserAct.ANYTHING_ELSE]#, UserAct.GOODBYE]

            if 'reqt' in self.goal:
                # #todo if info_answered, shouldn't ask_info again
                possible_actions.append(UserAct.ASK_INFO)
            elif 'cur_book' in self.goal:
                possible_actions.append(UserAct.MAKE_RESERVATION)


        selected_action = self.sample(possible_actions)

        params = None
        if selected_action == UserAct.INFORM_TYPE:
            params = {}
            for entity, value in self.goal['cur_info'].items():
                if restaurant_info[entity] != value:
                    params[entity] = value
                    # update state
                    self.state['informed'][entity] += 1

        elif selected_action == UserAct.ASK_INFO:
            params = {entity: None for entity in self.goal['reqt']}
            # update states
            for entity in self.goal['reqt']:
                self.state['asked'][entity] += 1
        elif selected_action == UserAct.MAKE_RESERVATION:
            params = self._generate_params(selected_action)

        elif selected_action == UserAct.ANYTHING_ELSE:
            params = None
            # update states
            self.state['asked_anything_else'] += 1



        return Action(selected_action, params)

    def response_NO_OTHER(self, sys_act):
        # no_other --> (ask_info, make_reservation)
        # can only be the response of ANYTHING_ELSE
        self.state['no_other_presented'] += 1

        self.check_constrain.append(self.check_presented_result(match=dialog_config.NO_OTHER))

        possible_actions = []
        if 'reqt' in self.goal:
            possible_actions.append(UserAct.ASK_INFO)
        elif 'cur_book' in self.goal:
            possible_actions.append(UserAct.MAKE_RESERVATION)

        selected_action = self.sample(possible_actions)

        params = None
        if selected_action == UserAct.ASK_INFO:
            params = {entity: None for entity in self.goal['reqt']}
            # update states
            for entity in self.goal['reqt']:
                self.state['asked'][entity] += 1
        elif selected_action == UserAct.MAKE_RESERVATION:
            params = self._generate_params(selected_action)
        else:
            raise ValueError("impossible action at the stage {}".format(selected_action))

        return Action(selected_action, params)

    def response_NOMATCH_RESULT(self, sys_act):
        # if no match --> goodbye or change_type
        # checked
        self.state['no_match_presented'] += 1

        if 'info_second_choice' in self.goal and self.state['no_match_presented'] == 1:
            possible_actions = [UserAct.INFORM_TYPE_CHANGE]
        else:
            possible_actions = [UserAct.GOODBYE]
        selected_action = self.sample(possible_actions)

        self.check_constrain.append(self.check_presented_result(match=dialog_config.NO_MATCH))

        if selected_action == UserAct.INFORM_TYPE_CHANGE:
            if self.state['no_match_presented'] == 1:
                # informed only once 'no_match', move on to the second choice
                params = {}
                for k in self.goal['info_second_choice']:
                    self.goal['cur_info'][k] = self.goal['info_second_choice'][k]
                    params[k] = self.goal['info_second_choice'][k]
        else:
            params = None

        return Action(selected_action, params)

    def response_PROVIDE_INFO(self, sys_act):
        # if provide_info --> goodbye
        # checked
        assert sys_act.act == SystemAct.PROVIDE_INFO

        # update states
        for entity, value in sys_act.parameters.items():
            if entity in self.entity_type['requestable_slots']:
                self.state['asked_answered'][entity] += 1

        self.check_info = self.check_provided_info(sys_act.parameters)
        possible_actions = [UserAct.GOODBYE]
        selected_action = self.sample(possible_actions)

        params = None
        if selected_action == UserAct.GOODBYE:
            params = None

        return Action(selected_action, params)

    def response_ASK_RESERVATION_INFO(self, sys_act):
        # shouldn't appear
        assert sys_act.act == SystemAct.ASK_RESERVATION_INFO
        self.check_reservation.append(self.check_reservation_result(sys_act.act))
        if self.check_reservation[-1] == dialog_config.RESERVATION_CHECK_FAILURE:
            possible_actions = [UserAct.GOODBYE]
        else:
            possible_actions = [UserAct.MAKE_RESERVATION]
        selected_action = self.sample(possible_actions)

        params = None
        if selected_action == UserAct.MAKE_RESERVATION:
            params = self._generate_params(selected_action, sys_act)
        return Action(selected_action, params)


    def response_BOOKING_SUCCESS(self, sys_act):
        # book_success --> goodbye
        # checked
        self.check_reservation.append(self.check_reservation_result(sys_act.act))
        possible_actions = [UserAct.GOODBYE]
        selected_action = self.sample(possible_actions)

        params = None
        if selected_action == UserAct.GOODBYE:
            params = None
        return Action(selected_action, params)

    def response_BOOKING_FAIL(self, sys_act):
        # book_success --> (goodbye, change_time)
        self.state['book_fail'] += 1
        self.check_reservation.append(self.check_reservation_result(sys_act.act))

        if 'book_second_choice' in self.goal and self.state['book_fail'] == 1:
            possible_actions = [UserAct.MAKE_RESERVATION_CHANGE_TIME]
        else:
            possible_actions = [UserAct.GOODBYE]

        selected_action = self.sample(possible_actions)

        if selected_action == UserAct.MAKE_RESERVATION_CHANGE_TIME:
            if self.state['book_fail'] == 1:
                # informed only once 'no_match', move on to the second choice
                params = {}
                for k in self.goal['book_second_choice']:
                    self.goal['cur_book'][k] = self.goal['book_second_choice'][k]
                    params[k] = self.goal['book_second_choice'][k]
        else:
            params = None

        return Action(selected_action, params)

    def check_pair(self, sys_act_str):
        last_usr_act = self.state['usr_act_sequence'][-1]
        if last_usr_act == UserAct.INFORM_TYPE:
            if sys_act_str not in [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.INFORM_TYPE_CHANGE:
            if sys_act_str not in [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.ASK_INFO:
            if sys_act_str not in [SystemAct.PROVIDE_INFO]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.MAKE_RESERVATION:
            if sys_act_str not in [SystemAct.ASK_RESERVATION_INFO, SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
            if sys_act_str not in [SystemAct.ASK_RESERVATION_INFO, SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.ANYTHING_ELSE:
            if sys_act_str not in [SystemAct.NO_OTHER, SystemAct.PRESENT_RESULT]:
                self.dialog_status = dialog_config.FAILED_DIALOG

        elif last_usr_act == UserAct.GOODBYE:
            if sys_act_str not in [SystemAct.GOODBYE]:
                self.dialog_status = dialog_config.FAILED_DIALOG




    def evaluate_GOOD_BYE(self, sys_act):
        # success conditions: 1) present correct restaurant 2) present correct info/ try to make a reservation
        # failure conditions: 1) check_constrain = FALSE (the result presented is incorrect)
        #                     2) didn't answer ask_info, i.e. value in self.state['asked'] > 0
        #
        if sys_act.act == SystemAct.GOODBYE and UserAct.GOODBYE not in self.state['usr_act_sequence']:
            self.dialog_status = dialog_config.FAILED_DIALOG
            return

        # 1. check the restaurant info
        if len(self.check_constrain) == 0:
            #didn't present result at all
            self.dialog_status = dialog_config.FAILED_DIALOG
            return
        else:
            # presented some results
            if UserAct.INFORM_TYPE_CHANGE in self.state['usr_act_sequence']:
                if len(self.check_constrain) < 2:
                    # because there is a second option,
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass
            elif UserAct.ANYTHING_ELSE in self.state['usr_act_sequence']:
                if len(self.check_constrain) < 2:
                    # because there is a second option,
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass
            else:
                # no second option
                all_constrain = [(c == dialog_config.CONSTRAINT_CHECK_SUCCESS) for c in self.check_constrain]
                if not np.all(all_constrain):
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    # need to check the last one in case of "inform_type_change"
                    pass

        if UserAct.ASK_INFO in self.state['usr_act_sequence']:
            if self.check_info == dialog_config.INFO_CHECK_NOTYET:
                print("INFO_CHECK_NOTYET")
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            elif self.check_info == dialog_config.INFO_CHECK_FAILURE:
                print("INFO_CHECK_FAILURE")
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            else:
                pass
                # self.dialog_status = dialog_config.SUCCESS_DIALOG

        if UserAct.MAKE_RESERVATION in self.state['usr_act_sequence']:
            if len(self.check_reservation) == 0:
                # didn't present result at all
                self.dialog_status = dialog_config.FAILED_DIALOG
                return
            else:
                all_reservation_constrain = [(c == dialog_config.RESERVATION_CHECK_SUCCESS) for c in self.check_reservation]
                if not np.all(all_reservation_constrain):
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return
                else:
                    pass

        self.dialog_status = dialog_config.SUCCESS_DIALOG
        return

        if self.check_constrain == dialog_config.CONSTRAINT_CHECK_FAILURE:
            print("CONSTRAINT_CHECK_FAILURE")
            self.dialog_status = dialog_config.FAILED_DIALOG
        elif self.check_constrain == dialog_config.CONSTRAINT_CHECK_NOTYET:
            print("CONSTRAINT_CHECK_NOTYET")
            self.dialog_status = dialog_config.FAILED_DIALOG
        else:
            if self.goal['first_choice_match'] > 0:
                if 'reqt' in self.goal:
                    if self.check_info == dialog_config.INFO_CHECK_NOTYET:
                        print("INFO_CHECK_NOTYET")
                        self.dialog_status = dialog_config.FAILED_DIALOG
                    elif self.check_info == dialog_config.INFO_CHECK_FAILURE:
                        print("INFO_CHECK_FAILURE")
                        self.dialog_status = dialog_config.FAILED_DIALOG
                    else:
                        self.dialog_status = dialog_config.SUCCESS_DIALOG
                elif 'cur_book' in self.goal:
                    if self.check_reservation == dialog_config.RESERVATION_CHECK_NOTYET:
                        print("RESERVATION_CHECK_NOTYET")
                        self.dialog_status = dialog_config.FAILED_DIALOG
                    elif self.check_reservation == dialog_config.RESERVATION_CHECK_FAILURE:
                        print("RESERVATION_CHECK_FAILURE")
                        self.dialog_status = dialog_config.FAILED_DIALOG
                    else:
                        self.dialog_status = dialog_config.SUCCESS_DIALOG
            else:
                self.dialog_status = dialog_config.SUCCESS_DIALOG

    def respond(self, sys_act=None):
        if sys_act is not None:
            self.state['sys_act_sequence'].append(sys_act.act)
        usr_act = self.rule_policy(sys_act=sys_act)
        self.state['usr_act_sequence'].append(usr_act.act)
        # print(usr_act)
        if self.nlg_templates:
            usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act, templates=self.nlg_templates, generator=1)
        else:
            usr_response_sent, lexicalized_usr_act = self.nlg.generate_sent(usr_act)
        return usr_act, usr_response_sent


"""
    def generate_one_goal_entity(self):

        goal = {'food': None,
                'area': None,
                'pricerange': None,
                'address': None,
                'postcode': None,
                'phone': None}

        goal['food'] = random.choice(self.entity_pool['food'], size=1)[0]
        goal['area'] = random.choice(self.entity_pool['area'], size=1)[0]
        goal['pricerange'] = random.choice(self.entity_pool['pricerange'], size=1)[0]
        goal['address'] = random.choice([0, 1], size=1)[0] # 0/1, ask for address or not
        goal['postcode'] = random.choice([0, 1], size=1)[0] # 0/1, ask or not
        goal['phone'] = random.choice([0, 1], size=1)[0] # 0/1, ask or not

        return goal

    def agenda_response(self, system_utt, dialog_state):
        queryable = dialog_state['informable_slots']['food'] and dialog_state['informable_slots']['area'] \
                    and dialog_state['informable_slots']['pricerange']
        if self.response_history:
            last_question_about_address = "slot_address" in self.response_history[-1] or "slot_postcode" in self.response_history[-1]\
                        or "slot_phone" in self.response_history
        if system_utt == "do you have a [slot_food] preference ?":
            if last_question_about_address and dialog_state['match_presented'] > 0:
                return ["ERROR, I asked for the information but you didn't answer me!"]
            else:
                if dialog_state['informable_slots']['food'] == 0:
                    return [4]
                else:
                    return ["MINOR_ERROR, you already know food"]
        elif system_utt == "do you have a [slot_area] preference ?":
            if last_question_about_address and dialog_state['match_presented'] > 0:
                return ["ERROR, I asked for the information but you didn't answer me!"]
            else:
                if dialog_state['informable_slots']['area'] == 0:
                    return [4]
                else:
                    return ["MINOR_ERROR, you already know area"]
        elif system_utt == "do you have a [slot_pricerange] preference ?":
            if last_question_about_address and dialog_state['match_presented'] > 0:
                return ["ERROR, I asked for the information but you didn't answer me!"]
            else:
                if dialog_state['informable_slots']['pricerange'] == 0:
                    return [4]
                else:
                    return ["MINOR_ERROR, you already know pricerange"]

        elif system_utt == \
                "I am sorry, but there are no restaurants matching your request. Is there anything else I can help you with?":
            if last_question_about_address and dialog_state['match_presented'] > 0:
                return ["ERROR, I asked for the information but you didn't answer me!"]
            else:
                if queryable: # queryable
                    if dialog_state['match_presented'] >= self.goal['match_nums'][0]:
                        if dialog_state['no_match_presented'] == 1:
                            return [0, 5, 8]
                        else:
                            return ["ERROR, I already know there is no match!"]
                    else:
                        return ["ERROR, there is at least one more match!"]
                else:
                    return ["ERROR, you don't have enough entities to make a query!"]

        elif system_utt == \
            "[value_name] is a good restaurant matching your request. Is there anything else I can help you with?":
            if last_question_about_address and dialog_state['match_presented'] > 0:
                return ["ERROR, I asked for the information but you didn't answer me!"]
            else:
                if queryable:
                    if dialog_state['match_presented'] <= self.goal['match_nums'][0]:
                        return [0, 1, 2, 3, 5, 8]
                    else:
                        return ["ERROR, you actually run out of results!"]
                else:
                    return ["ERROR, you don't have enough entities to make a query!"]

        elif system_utt == \
            "[value_name] is located at [value_address] . its [slot_phone] is [value_phone] and the [slot_postcode] is [value_postcode] . is there anything else i can help you with ?":
            if dialog_state['match_presented'] + dialog_state['no_match_presented'] > 0:
                asked_for_info = dialog_state['requestable_slots_asked']['address'] == 1 \
                     or dialog_state['requestable_slots_asked']['phone'] == 1 \
                     or dialog_state['requestable_slots_asked']['postcode'] == 1
                info_provided = dialog_state['requestable_slots_provided']['address'] == 1 \
                     or dialog_state['requestable_slots_provided']['phone'] == 1 \
                     or dialog_state['requestable_slots_provided']['postcode'] == 1
                if asked_for_info and info_provided:
                    return [0, 5, 8]
                elif not asked_for_info and info_provided:
                    return ["ERROR, I haven't asked for the information"]
                elif asked_for_info and not info_provided:
                    return ["ERROR, you haven provided the info"]
                else:
                    return ["ERROR, this is out of no where!"]
            else:
                return ["ERROR, you haven't presented any result yet!"]
        elif system_utt == "thank you for using our system, goodbye .":
            return ["ERROR, you are not supposed to end the conversation!"]
        return ["ERROR"]



    def sample_response(self, system_utt, dialog_state, available_sents):

        def get_sample_from_slots(ok_slots):
            possible_sents = []
            for s in ok_slots:
                possible_sents += available_sents[s]
            if possible_sents:
                return [random.choice(possible_sents, size=1)[0]]
            else:
                return self.agenda_response(system_utt, dialog_state)

        available_slots = available_sents.keys()
        queryable = dialog_state['informable_slots']['food'] and dialog_state['informable_slots']['area'] \
                    and dialog_state['informable_slots']['pricerange']
        if self.response_history:
            last_question_about_address = "slot_address" in self.response_history[-1] or "slot_postcode" in self.response_history[-1]\
                        or "slot_phone" in self.response_history
        if self.first_utt:
            possible_slots = [tmp_s for tmp_s in available_slots if "value_pricerange" in tmp_s\
                              or "value_area" in tmp_s or "value_food" in tmp_s]
            self.first_utt = False
            return get_sample_from_slots(possible_slots)
        else:
            if system_utt == "do you have a [slot_food] preference ?":
                if last_question_about_address and dialog_state['match_presented'] > 0:
                    return ["ERROR, I asked for the information but you didn't answer me!"]
                else:
                    if dialog_state['informable_slots']['food'] == 0:
                        possible_slots = [tmp_s for tmp_s in available_slots if "value_food" in tmp_s]
                        return get_sample_from_slots(possible_slots)
                    else:
                        return ["MINOR_ERROR, you already know food"]
            elif system_utt == "do you have a [slot_area] preference ?":
                if last_question_about_address and dialog_state['match_presented'] > 0:
                    return ["ERROR, I asked for the information but you didn't answer me!"]
                else:
                    if dialog_state['informable_slots']['area'] == 0:
                        possible_slots = [tmp_s for tmp_s in available_slots if "value_area" in tmp_s]
                        return get_sample_from_slots(possible_slots)
                    else:
                        return ["MINOR_ERROR, you already know area"]
            elif system_utt == "do you have a [slot_pricerange] preference ?":
                if last_question_about_address and dialog_state['match_presented'] > 0:
                    return ["ERROR, I asked for the information but you didn't answer me!"]
                else:
                    if dialog_state['informable_slots']['pricerange'] == 0:
                        possible_slots = [tmp_s for tmp_s in available_slots if "value_pricerange" in tmp_s]
                        return get_sample_from_slots(possible_slots)
                    else:
                        return ["MINOR_ERROR, you already know pricerange"]

            elif system_utt == \
                    "I am sorry, but there are no restaurants matching your request. Is there anything else I can help you with?":
                if last_question_about_address and dialog_state['match_presented'] > 0:
                    return ["ERROR, I asked for the information but you didn't answer me!"]
                else:
                    if queryable: # queryable
                        if dialog_state['match_presented'] >= self.goal['match_nums'][0]:
                            if dialog_state['no_match_presented'] == 1:
                                possible_slots = [tmp_s for tmp_s in available_slots if "end" in tmp_s]
                                return get_sample_from_slots(possible_slots)
                            else:
                                return ["ERROR, I already know there is no match!"]
                        else:
                            return ["ERROR, there is at least one more match!"]
                    else:
                        return ["ERROR, you don't have enough entities to make a query!"]

            elif system_utt == \
                "[value_name] is a good restaurant matching your request. Is there anything else I can help you with?":
                if last_question_about_address and dialog_state['match_presented'] > 0:
                    return ["ERROR, I asked for the information but you didn't answer me!"]
                else:
                    if queryable:
                        if dialog_state['match_presented'] <= self.goal['match_nums'][0]:
                            possible_slots = [tmp_s for tmp_s in available_slots if "end" in tmp_s or "else" in tmp_s \
                                              or "slot_address" in tmp_s or "slot_phone" in tmp_s or "slot_postcode" in tmp_s]
                            return get_sample_from_slots(possible_slots)
                            # return [0, 1, 2, 3, 5, 8]
                        else:
                            return ["ERROR, you actually run out of results!"]
                    else:
                        return ["ERROR, you don't have enough entities to make a query!"]

            elif system_utt == \
                "[value_name] is located at [value_address] . its [slot_phone] is [value_phone] and the [slot_postcode] is [value_postcode] . is there anything else i can help you with ?":
                if dialog_state['match_presented'] + dialog_state['no_match_presented'] > 0:
                    asked_for_info = dialog_state['requestable_slots_asked']['address'] == 1 \
                         or dialog_state['requestable_slots_asked']['phone'] == 1 \
                         or dialog_state['requestable_slots_asked']['postcode'] == 1
                    info_provided = dialog_state['requestable_slots_provided']['address'] == 1 \
                         or dialog_state['requestable_slots_provided']['phone'] == 1 \
                         or dialog_state['requestable_slots_provided']['postcode'] == 1
                    if asked_for_info and info_provided:
                        possible_slots = [tmp_s for tmp_s in available_slots if "end" in tmp_s]
                        return get_sample_from_slots(possible_slots)
                        # return [0, 5, 8]
                    elif not asked_for_info and info_provided:
                        return ["ERROR, I haven't asked for the information"]
                    elif asked_for_info and not info_provided:
                        return ["ERROR, you haven provided the info"]
                    else:
                        return ["ERROR, this is out of no where!"]
                else:
                    return ["ERROR, you haven't presented any result yet!"]
            elif system_utt == "thank you for using our system, goodbye .":
                return ["ERROR, you are not supposed to end the conversation!"]
            return ["ERROR"]


    def sample_from(self, last_state, dialog_state, system_utt):

        next_state = random.choice(10, size=1, p=self.transition_prob[last_state])[0]
        response = self.sample_from_one_turn(next_state, dialog_state, system_utt)

        self.states_series.append(next_state)
        return response

    def sample_from_one_turn(self, state, dialog_state, system_utt):
        # dialog_state = {
        # "informable_slots": {"food":,
        #                      "area":,
        #                      "pricerange"},
        # "requestable_slots": {"address":
        #                   "phone":
        #                   "postcode"}
        # "presented": (0, haven't presented yet; 1, match; -1, no match)
        # "left_query":
        # }
        informable_slots = ["food", "area", "pricerange"]
        if not self.is_sample:
            if self.first_utt:
                entity_bit = random.choice(2, size=3)
                entity_provide = ["value_"+informable_slots[i] for i, e in enumerate(entity_bit) if e]
                response = " ".join(["I am looking for a restaurant, "] + entity_provide)
                self.first_utt = False
            else:
                if state == 4:
                    response = ""
                    if "slot_food" in system_utt:
                        response = "value_food"
                    if "slot_area" in system_utt:
                        response = "value_area"
                    if "slot_pricerange" in system_utt:
                        response = "value_pricerange"
                else:
                    response = STATE_TO_SENTS[state]
                agenda_state = self.agenda_response(system_utt, dialog_state)
                if type(agenda_state[0]) is int:
                    # altered_response = response
                    if state not in agenda_state:
                        altered_response = "$altered$"
                        agenda_state_tmp = random.choice(agenda_state, size=1)[0]
                        if agenda_state_tmp == 4:
                            if "slot_food" in system_utt:
                                altered_response = "$altered$ value_food"
                            if "slot_area" in system_utt:
                                altered_response = "$altered$ value_area"
                            if "slot_pricerange" in system_utt:
                                altered_response = "$altered$ value_pricerange"
                        else:
                            altered_response = "$altered$ " + STATE_TO_SENTS[agenda_state_tmp]
                        response = altered_response
                    else:
                        response = "$non-altered$ " + response
                else: # error message
                    response = agenda_state[0]
                    if "MINOR_ERROR" in response:
                        self.done = False
                        self.success = None
                    else:
                        self.done = True
                        self.success = False
        else:
            available_sents = self.slots_by_state[state]
            response = self.sample_response(system_utt, dialog_state, available_sents)

            if type(response[0]) is int: # ran out of sample, went to agenda_response
                altered_response = "$altered, run out of sample$"
                agenda_state_tmp = random.choice(response, size=1)[0]
                if agenda_state_tmp == 4:
                    if "slot_food" in system_utt:
                        altered_response = "$altered, run out of sample$ value_food"
                    if "slot_area" in system_utt:
                        altered_response = "$altered, run out of sample$ value_area"
                    if "slot_pricerange" in system_utt:
                        altered_response = "$altered, run out of sample$ value_pricerange"
                else:#
                    altered_response = "$altered, run out of sample$ " + STATE_TO_SENTS[agenda_state_tmp]
                response = altered_response

            else:
                response = response[0]
                if "MINOR_ERROR" in response: # error message
                    self.done = False
                    self.success = None
                elif "ERROR" in response:
                #if "ERROR" in response:
                    self.done = True
                    self.success = False


        if ("thank" in response or "bye" in response) and self.success is None:
            if dialog_state['match_presented'] + dialog_state['no_match_presented'] > 0:
                self.success = True
                self.done = True
            else:
                self.success = False
                self.done = True
        return response

"""

if __name__ == "__main__":
    user = User()
    goal = Goal()


