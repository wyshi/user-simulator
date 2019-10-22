from __future__ import division, print_function
import numpy as np
# from util import oneHotLabel
import pickle
import os
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import simulator.dialog_config as dialog_config
from simulator.agent.core import UserAct, SystemAct, Action
# from config import Config
import pdb

# CONFIG = Config()

cur_dir = os.getcwd()

import pandas as pd
import json
data_dir = "data/multiwoz-master/data/multi-woz/"
with open("data/multiwoz-master/data/multi-woz/restaurant_db.json", "r") as fh:
    DB = json.load(fh)

class Enviroment(object):
    """
    dialog manager,
    state = [<d>, <a>, <time>, <uncovered_d>, <uncovered_a>, last_action_1_hot]
    """
    def __init__(self, user, system, verbose=True, config=None):
        # self.state_dim = state_dim
        # self.num_action = num_action
        # self.num_entity = num_entity
        self.DB = DB
        self.sys_action_cardinality = dialog_config.SYS_ACTION_CARDINALITY
        self.usr_action_cardinality = dialog_config.USER_ACT_CARDINALITY

        self.user_action = None
        self.last_agent_action = None
        self.user = user
        self.system = system
        if config is not None:
            self.config = config
            # pdb.set_trace()
        else:
            raise NotImplementedError
            # self.config = CONFIG
        # self.state_maintained = self.zero_state(self.user.goal['total_query'])
        self.done = False
        self.success = None

        self.verbose = verbose

        self.step_i = 0

        self.first_step = True

        # self.MAX_NUM_ELSE = MAX_NUM_ELSE

    def zero_state(self, total_query=1):
        zero_dialog_state = {
            "informable_slots": {"food": 0,
                                 "area": 0,
                                 "pricerange": 0,
                                 "name": 0},
            "requestable_slots_provided": {"address": 0,
                                            "phone": 0,
                                            "postcode": 0},
            "requestable_slots_asked": {"address": 0,
                                           "phone": 0,
                                           "postcode": 0},
            "reservation_slots_provided": {"num_people": 0,
                                           "time": 0,
                                           "day": 0},

            "match_presented": 0,
            "no_match_presented": 0,
            "num_else_so_far": 0,
            "total_query": total_query
        }

        return zero_dialog_state

    def reset(self, mode, mturk_res=None):
        # state = env.reset()
        self.done = False
        self.success = None
        self.first_step = True
        self.user.reset()
        self.system.reset()

        self.step_i = 0
        self.last_sys_act, self.last_sys_sent = None, None

        print("goal", self.user.goal)
        # first user sentence
        next_state = self.step_user(mode=mode, mturk_res=mturk_res)

        return next_state
        # if config.INTERACTIVE:
        #     self.last_usr_sent = input('Please respond: ')
        #     self.last_usr_act_true = None
        # else:
        #     self.last_usr_act_true, self.last_usr_sent = self.user.respond(sys_act=self.last_sys_act)
        # self.last_usr_act_pred = self.system.nlu(usr_sent=self.last_usr_sent, usr_act=self.last_usr_act_true, mode=mode)

    def queryable(self):
        if self.state_maintained['informable_slots']['food'] and self.state_maintained['informable_slots']['area'] and self.state_maintained['informable_slots']['pricerange']:
            return True
        else:
            return False

    def query_status(self):
        match_nums_list = self.user.goal['match_nums']# self.query_in_DB()
        if match_nums_list[0] <= (self.state_maintained['no_match_presented'] + self.state_maintained["match_presented"]):
            return 1, 0 # [no_match, match]
        else:
            return 0, 1 # [no_match, match]

    def query_in_DB(self):
        query_expr = []
        for g in self.user.goal['goal_entity']:
            tmp_food_query = ""
            tmp_area_query = ""
            tmp_price_query = ""
            if g['food'] != "dontcare":
                tmp_food_query = " food == \"" + g['food'] + "\""
            if g['area'] != "dontcare":
                tmp_area_query = " area == \"" + g['area'] + "\""
            if g['pricerange'] != "dontcare":
                tmp_price_query = " pricerange == \"" + g['pricerange'] + "\""

            tmp_query = [tmp_food_query, tmp_area_query, tmp_price_query]
            tmp_query = [q for q in tmp_query if q]
            tmp_query = " and ".join(tmp_query)
            query_expr.append(tmp_query)

        match_nums = []
        for q in query_expr:
            if q:
                sample_from_subset = self.DB.query(q)
                match_nums.append(sample_from_subset.shape[0])

            else:
                match_nums.append(sample_from_subset.shape[0])

        return match_nums

    def maintain_states(self, who):
        """
        state = [<d>, <a>, <time>, <uncovered_d>, <uncovered_a>, last_action_1_hot]
        :param last_agent_action:
        :return:
        """
        self.state_maintained['total_query'] = self.user.goal['total_query']

        if who == "user":
            # entity status, user side
            if "value_food" in self.user_action: #TODO: value_doncare???
                self.state_maintained['informable_slots']['food'] = 1
            if "value_pricerange" in self.user_action:
                self.state_maintained['informable_slots']['pricerange'] = 1
            if "value_area" in self.user_action:
                self.state_maintained['informable_slots']['area'] = 1

            if "slot_postcode" in self.user_action:
                self.state_maintained['requestable_slots_asked']['postcode'] = 1
            if "slot_phone" in self.user_action:
                self.state_maintained['requestable_slots_asked']['phone'] = 1
            if "slot_address" in self.user_action:
                self.state_maintained['requestable_slots_asked']['address'] = 1

        elif who == "agent":
            if self.last_agent_action:
                # entity status, agent side
                if "value_postcode" in self.last_agent_action:
                    self.state_maintained['requestable_slots_provided']['postcode'] = 1
                if "value_phone" in self.last_agent_action:
                    self.state_maintained['requestable_slots_provided']['phone'] = 1
                if "value_address" in self.last_agent_action:
                    self.state_maintained['requestable_slots_provided']['address'] = 1

                # restaurant presented status
                if "[value_name] is a good restaurant" in self.last_agent_action:
                    self.state_maintained['match_presented'] += 1
                if "no restaurants matching your request" in self.last_agent_action:
                    self.state_maintained['no_match_presented'] += 1

    def update_state(self, act, who):

        if who == 'usr':
            self.system.update_state(act=act, who='usr')
        elif who == 'sys':
            self.system.update_state(act=act, who='sys')
        else:
            raise ValueError("{} is not allowed".format(who))

    def step(self, provided_sys_act=None, mode=dialog_config.RL_TRAINING):
        """
        next_state, reward, done, _ = env.step(action)
        :param action:
        :return:
        """
        result_step_sys = self.step_system(provided_sys_act=provided_sys_act, mode=mode)
        if result_step_sys is not None:
            # goes into FAILED_DIALOG, shouldn't happen in rule_policy and INTERACTIVE mode
            next_state, reward, self.done = result_step_sys
            print('reward per turn', reward)
            return next_state, reward, self.done

        next_state, reward, self.done = self.step_user(mode=mode)
        print('reward per turn', reward)
        return next_state, reward, self.done
        ############################################################################

    def step_user(self, mode, mturk_res=None):
        import pdb
        # pdb.set_trace()
        # first user sentence
        if self.config.INTERACTIVE:
            if mturk_res is None:
                self.last_usr_sent = input('Please respond: ')
            else:
                self.last_usr_sent = mturk_res
            self.last_usr_act_true = None
        else:
            self.last_usr_act_true, self.last_usr_sent = self.user.respond(sys_act=self.last_sys_act, prev_sys=self.last_sys_sent)

        if self.last_usr_act_true is None:
            # self.last_usr_act_true is None only when using SL-based simulator
            self.last_usr_act_pred = self.system.nlu(usr_sent=self.last_usr_sent, usr_act=self.last_usr_act_true, mode=dialog_config.RL_TRAINING)
        else:
            self.last_usr_act_pred = self.system.nlu(usr_sent=self.last_usr_sent, usr_act=self.last_usr_act_true, mode=mode)

        self.update_state(act=self.last_usr_act_pred, who='usr')

        if self.verbose and (not self.config.INTERACTIVE):
            print("{} Usr: {}".format(self.step_i, self.last_usr_sent))
            # print("True user act: ", self.last_usr_act_true)

        # if not self.config.use_sequicity_for_rl_model:
        #     next_state = self.system.prepare_state_representation()
        # else:
        #     next_state = None
        next_state = None

        if self.first_step:
            self.first_step = False
            return next_state

        if not self.config.INTERACTIVE:
            # next_state = self.system.prepare_state_representation()
            # if self.config.use_new_reward:
            #     reward = self.evaluate_cur_move_new()
            # else:
            reward = self.evaluate_cur_move()
            # reward = 0
            return next_state, reward, self.done
        else:
            # next_state = self.system.prepare_state_representation()
            reward = 0

            if self.last_usr_act_pred.act == UserAct.GOODBYE:
                self.done = True
                self.last_sys_act, self.last_sys_sent = Action(SystemAct.GOODBYE,
                                                               None), "Thanks for using the system! Have a good day!"
                if self.verbose:
                    print("{} Sys: {}".format(self.step_i, self.last_sys_sent))

            return next_state, reward, self.done

    def step_system(self, provided_sys_act=None, mode=dialog_config.RL_TRAINING):
        # print("in step_system: ", provided_sys_act)
        self.last_sys_act, self.last_sys_sent = self.system.respond(usr_act=self.last_usr_act_pred,
                                                                    usr_sent=self.last_usr_sent, mode=mode)
        # if mode == dialog_config.RL_TRAINING or mode == dialog_config.RANDOM_ACT:
        #     assert provided_sys_act is not None
        #     self.last_sys_act, self.last_sys_sent = self.system.respond(usr_utt=self.last_usr_sent)
        # else:
        #     # assert provided_sys_act is None
        #     self.last_sys_act, self.last_sys_sent = self.system.respond(provided_sys_act=provided_sys_act, mode=mode,
        #                                                                 usr_act=self.last_usr_act_pred, usr_sent=None)
        # if self.system.dialog_status == dialog_config.FAILED_DIALOG:
        #     next_state = None
        #     reward = dialog_config.FAILURE_REWARD
        #     self.done = True
        #     self.success = False
        #     return next_state, reward, self.done
        #
        self.update_state(act=self.last_sys_act, who='sys')
        #
        if self.verbose:
            print("{} Sys: {}".format(self.step_i, self.last_sys_sent))

        # user's next response to the current system act
        self.step_i += 1

        return None

    def evaluate_cur_move(self):

        # success condition
        # pdb.set_trace()
        # print('user dialog_status, ', self.user.dialog_status)
        # pdb.set_trace()
        if self.user.dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = dialog_config.SUCCESS_REWARD
            self.done = True
            self.success = True

        # failure condition
        elif self.user.dialog_status == dialog_config.FAILED_DIALOG:
            reward = dialog_config.FAILURE_REWARD
            self.done = True
            self.success = False

        # not done yet
        elif self.user.dialog_status in [dialog_config.NO_OUTCOME_YET, dialog_config.TURN_FAIL_FOR_SL, dialog_config.TURN_SUCCESS_FOR_SL]:
            reward = dialog_config.PER_TURN_REWARD
            self.done = False
            self.success = None

        return reward

    def evaluate_cur_move_new(self):
        # pdb.set_trace()


        def calculate_per_turn_reward(self, last_usr_act_str, sys_act_str):
            # last_usr_act = self.user.state['usr_act_sequence'][-2]
            turn_reward = dialog_config.PER_TURN_REWARD
            if last_usr_act_str == UserAct.INFORM_TYPE:
                if sys_act_str in [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.INFORM_TYPE_CHANGE:
                if sys_act_str in [SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.ASK_INFO:
                if sys_act_str in [SystemAct.PROVIDE_INFO]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.MAKE_RESERVATION:
                if sys_act_str in [SystemAct.ASK_RESERVATION_INFO, SystemAct.BOOKING_SUCCESS,
                                       SystemAct.BOOKING_FAIL]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.MAKE_RESERVATION_CHANGE_TIME:
                if sys_act_str in [SystemAct.BOOKING_SUCCESS,
                                       SystemAct.BOOKING_FAIL]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.ANYTHING_ELSE:
                if sys_act_str in [SystemAct.NO_OTHER, SystemAct.PRESENT_RESULT, SystemAct.ASK_TYPE,
                                       SystemAct.NOMATCH_RESULT]:
                    turn_reward = dialog_config.TURN_AWARD

            elif last_usr_act_str == UserAct.GOODBYE:
                if sys_act_str in [SystemAct.GOODBYE]:
                    turn_reward = dialog_config.TURN_AWARD

            # if config.use_repetition_penalty:
            #     # repetition penalty
            #     informed_so_far = [len(value) > 0 for entity, value in self.system.state['informed'].items() if entity != 'name']
            #
            #     if np.all(informed_so_far) and sys_act_str == SystemAct.ASK_TYPE:
            #         turn_reward = dialog_config.repetition_penalty
            return turn_reward

        # success condition
        if self.user.dialog_status == dialog_config.SUCCESS_DIALOG:
            reward = dialog_config.SUCCESS_REWARD
            self.done = True
            self.success = True

        # failure condition
        elif self.user.dialog_status == dialog_config.FAILED_DIALOG:
            reward = dialog_config.FAILURE_REWARD
            self.done = True
            self.success = False

        # not done yet
        elif self.user.dialog_status in [dialog_config.NO_OUTCOME_YET, dialog_config.TURN_FAIL_FOR_SL, dialog_config.TURN_SUCCESS_FOR_SL]:
            reward = calculate_per_turn_reward(self, last_usr_act_str=self.user.state['usr_act_sequence'][-2],
                                               sys_act_str=self.last_sys_act.act)
            self.done = False
            self.success = None

        return reward

