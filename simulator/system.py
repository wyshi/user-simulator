import numpy as np
import json
from simulator.agent.core import Action, SystemAct, UserAct
from simulator.nlg import SysNlg
import simulator.dialog_config as dialog_config
from simulator.utils import oneHotLabel
import pickle as pkl
from simulator.multiwoz_utils import delexicalize
from simulator.nlu_model.main_nlu_train import load_nlu_model, single_pred
#from config import Config

#config = Config()

data_dir = "data/multiwoz-master/data/multi-woz/"
with open("data/multiwoz-master/data/multi-woz/restaurant_db.json", "r") as fh:
    DB = json.load(fh)

with open("data/multiwoz-master/data/multi-woz/delex_dic.pkl", "rb") as fh:
    DELEX_DIC = pkl.load(fh)


class System(object):
    def __init__(self, config):
        self.action_history = []
        # self.env = env
        self.DB = DB
        self.with_bit = False
        self.config = config
        self.sys_action_cardinality = dialog_config.SYS_ACTION_CARDINALITY

        self._set_nlu_model()
        self.nlg = SysNlg(domain=None, complexity=None)
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        self.initialize_episodes()

    def initialize_episodes(self):
        self._set_initial_state()
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        # self._set_initial_book_result()

    def _set_nlu_model(self):
        self.nlu_model = load_nlu_model(self.config.rule_base_sys_nlu)

    def _set_initial_state(self):
        self.state = {
            'informed': {k: [] for k in dialog_config.informable_slots},
            'asked': {k: [] for k in dialog_config.requestable_slots},
            'asked_answered': {k: [] for k in dialog_config.requestable_slots},

            'reservation_informed': {k: [] for k in dialog_config.reservation_slots},

            'results': [],
            'match_presented': 0,
            'no_match_presented': 0,
            'asked_anything_else': 0,
            'no_other_presented': 0,
            'book_fail': 0,
            'book_success': 0,

            'usr_act_sequence': [],
            'sys_act_sequence': [],

            'usr_act_par_sequence': [],
            'sys_act_par_sequence': []

        }

        # self.check_constrain = dialog_config.CONSTRAINT_CHECK_NOTYET
        # self.check_info = dialog_config.INFO_CHECK_NOTYET
        # self.check_reservation = dialog_config.RESERVATION_CHECK_NOTYET
        # self.dialog_status = dialog_config.NO_OUTCOME_YET

    def _set_initial_book_result(self):
        self._book_result = [self.sample([True, False]), self.sample([True, False])]

    def sample(self, examples, n=1, p=None):
        if p is None:
            # uniform
            p = [1 / len(examples)] * len(examples)

        if n == 1:
            return np.random.choice(examples, p=p)
        else:
            return list(np.random.choice(examples, n, replace=False, p=p))

    def reset(self):
        """
        <d>, <a>, <time>, <uncovered_d>, <uncovered_a>
        self.user_action = self.user.reset()
        :return:
        """
        self.initialize_episodes()
        self.action_history = []

    def query_in_DB(self, cur_info, skip=[]):
        match_list = []
        if len(skip):
            restaurants_to_skip = [(r['name'], r['food'], r['pricerange'], r['area']) for r in skip]
        for restaurant in self.DB:
            if len(skip):
                rest_info = (restaurant['name'], restaurant['food'], restaurant['pricerange'], restaurant['area'])
                if rest_info in restaurants_to_skip:
                    continue
            match = True
            for entity, value in cur_info.items():
                if restaurant[entity] != value:
                    match = False
            if match:
                match_list.append(restaurant)
        return match_list

    def update_state(self, act, who, mode=None):
        if who == 'usr':

            self._update_state_usr(act, mode=mode)

        elif who == 'sys':
            if act.act == SystemAct.ASK_TYPE:
                pass
            elif act.act == SystemAct.PRESENT_RESULT:
                self.state['results'].append(act.parameters)
                self.state['match_presented'] += 1
            elif act.act == SystemAct.NOMATCH_RESULT:
                self.state['no_match_presented'] += 1
            elif act.act == SystemAct.NO_OTHER:
                self.state['no_other_presented'] += 1
            elif act.act == SystemAct.PROVIDE_INFO:
                for entity, value in act.parameters.items():
                    if entity in dialog_config.requestable_slots:
                        self.state['asked_answered'][entity].append(value)
            elif act.act == SystemAct.BOOKING_SUCCESS:
                self.state['book_success'] += 1
            elif act.act == SystemAct.BOOKING_FAIL:
                self.state['book_fail'] += 1
            elif act.act == SystemAct.GOODBYE:
                pass

            self.state['sys_act_sequence'].append(act.act)
            self.state['sys_act_par_sequence'].append(act)


        else:
            raise ValueError("who disallowed {}".format(who))

    def _update_state_usr(self, act, mode=None):
        params = act.parameters

        # rule: if last turn
        if len(self.state['sys_act_par_sequence']) > 0 and self.state['sys_act_par_sequence'][-1].act == SystemAct.ASK_TYPE:
            for asked_entity_last_turn in self.state['sys_act_par_sequence'][-1].parameters:
                if asked_entity_last_turn not in params:
                    if len(self.state['informed'][asked_entity_last_turn]) == 0:
                        self.state['informed'][asked_entity_last_turn].append(dialog_config.I_DO_NOT_CARE)
        # rule: the same with reserve_info
        if len(self.state['sys_act_par_sequence']) > 0 and self.state['sys_act_par_sequence'][-1].act == SystemAct.ASK_RESERVATION_INFO:
            for asked_entity_last_turn in self.state['sys_act_par_sequence'][-1].parameters:
                if asked_entity_last_turn not in params:
                    #if len(self.state['reservation_informed'][asked_entity_last_turn]) == 0:
                    # unlike in the previous case, here we should be more broad, because the value of reservation slots won't impact
                    # the reservation results
                    self.state['reservation_informed'][asked_entity_last_turn].append(dialog_config.I_DO_NOT_CARE)

        # update parameters
        for entity, value in params.items():
            if entity in dialog_config.informable_slots:
                if value is not None:
                    self.state['informed'][entity].append(value)
            if entity in dialog_config.requestable_slots:
                if value is None:
                    self.state['asked'][entity].append(value)
            if entity in dialog_config.reservation_slots:
                if value is not None:
                    if entity == "people":
                        # if act.act in [UserAct.MAKE_RESERVATION, UserAct.MAKE_RESERVATION_CHANGE_TIME]:
                        self.state['reservation_informed'][entity].append(value)
                    else:
                        self.state['reservation_informed'][entity].append(value)

        # update acts
        if act.act == UserAct.INFORM_TYPE:
            pass

        elif act.act == UserAct.INFORM_TYPE_CHANGE:
            self.state['match_presented'] = 0


        elif act.act == UserAct.ASK_INFO:
            pass

        elif act.act == UserAct.MAKE_RESERVATION:
            pass

        elif act.act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
            pass

        elif act.act == UserAct.ANYTHING_ELSE:
            self.state['asked_anything_else'] += 1

        elif act.act == UserAct.GOODBYE:
            pass

        self.state['usr_act_sequence'].append(act.act)
        self.state['usr_act_par_sequence'].append(act)

        if False:
            # the code below can be used only if it's using rule-based user simulator
            if act.act == UserAct.INFORM_TYPE:
                for entity, value in act.parameters.items():
                    self.state['informed'][entity].append(value)

            elif act.act == UserAct.INFORM_TYPE_CHANGE:
                self.state['match_presented'] = 0
                for entity, value in act.parameters.items():
                    self.state['informed'][entity].append(value)

            elif act.act == UserAct.ASK_INFO:
                for entity, value in act.parameters.items():
                    self.state['asked'][entity].append(value)

            elif act.act == UserAct.MAKE_RESERVATION:
                for entity, value in act.parameters.items():
                    self.state['reservation_informed'][entity].append(value)

            elif act.act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
                for entity, value in act.parameters.items():
                    self.state['reservation_informed'][entity].append(value)

            elif act.act == UserAct.ANYTHING_ELSE:
                self.state['asked_anything_else'] += 1

            elif act.act == UserAct.GOODBYE:
                pass

            self.state['usr_act_sequence'].append(act.act)
            self.state['usr_act_par_sequence'].append(act)

    def respond(self, provided_sys_act=None, mode=dialog_config.RL_TRAINING, usr_act=None, usr_sent=None):
        """

        :param usr_sent:
        :param provided_sys_act: should be an integer
        :param warm_start:
        :param usr_act:
        :return:
        """
        # print("in respond: ", provided_sys_act)
        assert usr_act
        if mode == dialog_config.RL_WARM_START:
            assert usr_act # use ground truth usr_act
            assert provided_sys_act is None # use rule_policy
        elif mode == dialog_config.RL_TRAINING:
            # usr_act = self.nlu(usr_sent) # use predicted usr_act
            assert provided_sys_act is not None # use RL_policy
        elif mode == dialog_config.RANDOM_ACT:
            assert usr_act
            assert provided_sys_act is not None
        elif mode == dialog_config.INTERACTIVE:
            # assert provided_sys_act is None
            pass

        if provided_sys_act is not None:
            print("using provided sys act")
            sys_act = self._index_to_action(provided_sys_act, usr_act=usr_act)
        else:
            print("using rule_policy")
            sys_act = self.rule_policy(usr_act=usr_act)

        if self.dialog_status != dialog_config.FAILED_DIALOG:
            sys_response_sent, lexicalized_sys_act = self.nlg.generate_sent(sys_act)
        else:
            sys_response_sent, lexicalized_sys_act = None, None
        return sys_act, sys_response_sent

    def rule_policy(self, usr_act):
        if usr_act.act == UserAct.INFORM_TYPE:
            sys_act = self._response_INFORM_TYPE(usr_act)

        elif usr_act.act == UserAct.INFORM_TYPE_CHANGE:
            sys_act = self._response_INFORM_TYPE_CHANGE(usr_act)

        elif usr_act.act == UserAct.ASK_INFO:
            sys_act = self._response_ASK_INFO(usr_act)

        elif usr_act.act == UserAct.MAKE_RESERVATION:
            sys_act = self._response_MAKE_RESERVATION(usr_act)

        elif usr_act.act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
            sys_act = self._response_MAKE_RESERVATION_CHANGE_TIME(usr_act)

        elif usr_act.act == UserAct.ANYTHING_ELSE:
            sys_act = self._response_ANYTHING_ELSE()

        elif usr_act.act == UserAct.GOODBYE:
            sys_act = self._response_GOODBYE()

        return sys_act
        """
        queryable = ((len(self.state['informed']['pricerange']) > 0) and (len(self.state['informed']['area']) > 0) and (len(self.state['informed']['food']) > 0)) \
                    or (len(self.state['informed']['name']) > 0)
        if not queryable:
            assert usr_act.act == UserAct.INFORM_TYPE
            # the only option is to ask_type until all the entities are collected
            possible_actions = [SystemAct.ASK_TYPE]
            selected_action = self.sample(possible_actions)

            if selected_action == SystemAct.ASK_TYPE:
                params = {}
                for entity, value in self.state['informed'].items():
                    if entity != 'name':
                        if len(value) == 0:
                            params[entity] = None
            else:
                raise ValueError("disallowed sys_Act {}".format(selected_action))

            return Action(selected_action, params)
        else:


        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)
        """

    def nlu(self, usr_sent, usr_act=None, mode=None):

        if (mode == dialog_config.RANDOM_ACT) or (mode == dialog_config.RL_WARM_START):
            # return the true usr_act
            delex_sent, kv_dic = delexicalize.delexicalize_one_sent(usr_sent)
            self.last_usr_sent = delex_sent
            return usr_act
        elif mode == dialog_config.RL_TRAINING or mode == dialog_config.INTERACTIVE:
            # return usr_act
            # print("Warning: temporarily use the real user_act")
            delex_sent, kv_dic = delexicalize.delexicalize_one_sent(usr_sent)

            params = {}
            for k, v in kv_dic.items():
                if ('value_area') in k or ('value_place') in k:
                    params['area'] = v
                elif 'value_food' in k:
                    params['food'] = v
                elif 'value_pricerange' in k:
                    params['pricerange'] = v
                elif 'restaurant_name' in k:
                    params['name'] = v

                elif 'value_day' in k:
                    params['day'] = v
                elif 'value_time' in k:
                    params['time'] = v
                elif 'value_count' in k:
                    params['people'] = v
                else:
                    continue

            usr_act_str = single_pred(self.nlu_model, delex_sent)[0].lower()

            # import pdb
            # pdb.set_trace()
            if len(self.state['results']) == 0:
                if self.config.use_sl_simulator:
                    usr_act_str = UserAct.INFORM_TYPE
                else:
                    usr_act_str = UserAct.INFORM_TYPE
                # pass
            if usr_act_str == UserAct.INFORM_TYPE:
                pass
            elif usr_act_str == UserAct.INFORM_TYPE_CHANGE:
                pass
            elif usr_act_str == UserAct.ASK_INFO:
                if "phone" in usr_sent:
                    params.update({"phone": None})
                if "address" in usr_sent:
                    params.update({'address': None})
                if "postcode" in usr_sent or "post code" in usr_sent:
                    params.update({'postcode': None})
                # params.update({'phone': None})
                # if "price" in usr_sent:
                #     params.update({'price': None})
                # if "area" in usr_sent:
                #     params.update({'area': None})
                # if "food" in usr_sent or "cuisine" in usr_sent or "type" in usr_sent:
                #     params.update({'food': None})
            elif usr_act_str == UserAct.ANYTHING_ELSE:
                pass
            elif usr_act_str == UserAct.MAKE_RESERVATION:
                pass
            elif usr_act_str == UserAct.MAKE_RESERVATION_CHANGE_TIME:
                pass
            elif usr_act_str == UserAct.GOODBYE:
                pass
            else:
                usr_act_str = UserAct.INFORM_TYPE


            usr_act = Action(usr_act_str, params)
            self.last_usr_sent = delex_sent
            print(usr_act)
            return usr_act

    def action_to_index(self, sys_act_str):
        """
        for external use
        :param sys_act:
        :return:
        """
        # sys_action_cardinality = max(dialog_config.action_to_index_dict.values()) + 1
        if sys_act_str in dialog_config.action_to_index_dict:
            return dialog_config.action_to_index_dict[sys_act_str]#, self.sys_action_cardinality
        else:
            return -1#, self.sys_action_cardinality

    def _index_to_action(self, sys_act_idx, usr_act=None):
        assert isinstance(sys_act_idx, (int, np.integer))
        if sys_act_idx == 1:
            # present result
            queryable = ((len(self.state['informed']['pricerange']) > 0) and (
                        len(self.state['informed']['area']) > 0) and (len(self.state['informed']['food']) > 0)) \
                        or (len(self.state['informed']['name']) > 0)
            if self.with_bit:
                assert queryable
            else:
                if not queryable:
                    print("cannot make a query now! missing slots")
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return None
                else:
                    return self.query_and_decide_present_mode(usr_act)

        elif sys_act_idx == 3:
            # book success/failure
            reservable = [len(value) for entity, value in self.state['reservation_informed'].items()]
            reservable = np.all(reservable)

            if self.with_bit:
                assert reservable
            else:
                if not reservable:
                    print("cannot make a reservation now! missing slots")
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return None
                else:
                    reservation_available = self.sample([True, False], p=[0.8, 0.2])
                    if reservation_available:
                        params = self._generate_params(SystemAct.BOOKING_SUCCESS)
                        return Action(SystemAct.BOOKING_SUCCESS, params)
                    else:
                        return Action(SystemAct.BOOKING_FAIL, None)
        else:
            sys_act_str = dialog_config.index_to_action_dict[sys_act_idx]
            params = self._generate_params(sys_act_str=sys_act_str, usr_act=usr_act)

            return Action(sys_act_str, params)
            # if act == SystemAct.ASK_TYPE:
            #     params = self._generate_params(act)
            #     return Action(act, params)
            # elif act == SystemAct.PROVIDE_INFO:
            #     params = self._gen

    def query_and_decide_present_mode(self, usr_act):
        if len(self.state['informed']['name']) > 0:
            # specific restaurant
            cur_info = {'name': self.state['informed']['name'][-1]}
        else:
            cur_info = {entity: self.state['informed'][entity][-1] for entity in ['pricerange', 'area', 'food'] \
                        if self.state['informed'][entity][-1] != dialog_config.I_DO_NOT_CARE}
        match_result = self.query_in_DB(cur_info, skip=self.state['results'])
        if len(match_result) > 0:
            present_result = match_result[0]
            params = present_result  # {present_result[entity] for entity in dialog_config.informable_slots}
            return Action(SystemAct.PRESENT_RESULT, params)
        else:
            if usr_act.act == UserAct.ANYTHING_ELSE and len(match_result) == 0:
                return Action(SystemAct.NO_OTHER, None)
            else:
                return Action(SystemAct.NOMATCH_RESULT, None)

    def prepare_state_representation(self):
        """
        self.state = {
            'informed': {k: [] for k in dialog_config.informable_slots},
            'asked': {k: [] for k in dialog_config['requestable_slots']},
            'reservation_informed': {k: [] for k in dialog_config.reservation_slots},

            'asked_answered': {k: [] for k in dialog_config['requestable_slots']},


            'results': [],
            'match_presented': 0,
            'no_match_presented': 0,
            'asked_anything_else': 0,
            'no_other_presented': 0,
            'book_fail': 0,
            'book_success': 0,

            'usr_act_sequence': [],
            'sys_act_sequence': []

        }

        :param state:
        :return:
        """
        # print('state in system.prepare', self.state)
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        last_usr_act_rep = oneHotLabel(dialog_config.USER_ACTION_TO_INDEX[self.state['usr_act_sequence'][-1]],
                                       dim=dialog_config.USER_ACT_CARDINALITY)

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = [len(value) for entity, value in self.state['informed'].items()]

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_ask_slots_rep = [len(value) for entity, value in self.state['asked'].items()]

        ########################################################################
        #   Create bag of reserve slots representation to represent the current user action
        ########################################################################
        user_reserve_slots_rep = [len(value) for entity, value in self.state['reservation_informed'].items()]


        ########################################################################
        #   Encode last agent act
        ########################################################################
        if len(self.state['sys_act_sequence']) > 0:
            sys_act_index = self.action_to_index(self.state['sys_act_sequence'][-1])
            last_sys_act_rep = oneHotLabel(sys_act_index, dim=dialog_config.SYS_ACTION_CARDINALITY)
        else:
            # sys_act_index, sys_act_cardinality = self.action_to_index("<START>")
            last_sys_act_rep = oneHotLabel(-1, dim=dialog_config.SYS_ACTION_CARDINALITY)

        ########################################################################
        #   Creat bag of asked_answered slots based on the current_slots
        ########################################################################
        sys_ask_answered_slots_rep = [len(value) for entity, value in self.state['asked_answered'].items()]

        ########################################################################
        #   Creat bag of numerical slots based on the current_slots
        ########################################################################

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        KB_results_count = len(self.state['results'])
        match_presented_count = self.state['match_presented']
        no_match_presented_count = self.state['no_match_presented']
        no_other_presented_count = self.state['no_other_presented']
        asked_anything_else_count = self.state['asked_anything_else']
        book_fail_count = self.state['book_fail']
        book_success_count = self.state['book_success']

        # ########################################################################
        # #   Encode last agent inform slots
        # ########################################################################
        # agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        # if agent_last:
        #     for slot in agent_last['inform_slots'].keys():
        #         agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0
#
        # ########################################################################
        # #   Encode last agent request slots
        # ########################################################################
        # agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        # if agent_last:
        #     for slot in agent_last['request_slots'].keys():
        #         agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
#
        # turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        # turn_onehot_rep = np.zeros((1, self.max_turn))
        # turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        #kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        #for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
#
        #########################################################################
        ##   Representation of KB results (binary)
        #########################################################################
        #kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
        #    kb_results_dict['matching_all_constraints'] > 0.)
        #for slot in kb_results_dict:
        #    if slot in self.slot_set:
        #        kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack(
            [last_usr_act_rep, user_inform_slots_rep, user_ask_slots_rep, user_reserve_slots_rep,
             last_sys_act_rep, sys_ask_answered_slots_rep,
             KB_results_count, match_presented_count, no_match_presented_count, no_other_presented_count, asked_anything_else_count, book_success_count, book_fail_count])
        # print('in prepare_representation, in system.py, last usr sent', self.last_usr_sent)
        self.final_representation = [self.final_representation, self.last_usr_sent]

        return self.final_representation

    def _generate_params(self, sys_act_str, usr_act=None):
        """
        SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.PROVIDE_INFO,
                                       SystemAct.BOOKING_SUCCESS
        :param sys_act:
        :return:
        """

        if sys_act_str == SystemAct.ASK_TYPE:
            params = {}
            for entity, value in self.state['informed'].items():
                if entity != 'name':
                    if len(value) == 0:
                        params[entity] = None
            if params == {}:
                print("Warning: The system has already collected all the entities, shouldn't ask for info again!")
                random_select_entity = self.sample([entity for entity in self.state['informed'].keys() if entity != 'name'])
                params[random_select_entity] = params
        elif sys_act_str == SystemAct.PRESENT_RESULT:
            pass
        elif sys_act_str == SystemAct.PROVIDE_INFO:
            assert usr_act
            if self.with_bit:
                assert len(self.state['results']) > 0
            else:
                if len(self.state['results']) == 0:
                    print("haven't presented result yet!")
                    self.dialog_status = dialog_config.FAILED_DIALOG
                    return None
                else:
                    restaurant = self.state['results'][-1]

            params = {}
            # for entity in usr_act.parameters:
            #     if entity in restaurant:
            #         params[entity] = restaurant[entity]
            for entity, value in self.state['asked'].items():
                if len(value) > 0:
                    if entity in restaurant:
                        params[entity] = restaurant[entity]

            params['name'] = restaurant['name']
            for info in restaurant:
                params[info] = restaurant[info]

        elif sys_act_str == SystemAct.BOOKING_SUCCESS:
            params = {'reference': "ABC"}
        elif sys_act_str == SystemAct.ASK_RESERVATION_INFO:
            params = {entity: None for entity, value in self.state['reservation_informed'].items() if len(value) == 0}
            if len(params) == 0:
                print("Warning: the user has provided all the reservation info, but the system still asks")
                params = {self.sample(list(self.state['reservation_informed'].keys())): None}
        else:
            params = None
            print("Warning: the act {} doesn't take parameters".format(sys_act_str))

        return params

    def _response_INFORM_TYPE(self, usr_act):
        queryable = ((len(self.state['informed']['pricerange']) > 0) and (len(self.state['informed']['area']) > 0) and (len(self.state['informed']['food']) > 0)) \
                    or (len(self.state['informed']['name']) > 0)
        if not queryable:
            # the only optio.n is to ask_type until all the entities are collected
            possible_actions = [SystemAct.ASK_TYPE]
            selected_action = self.sample(possible_actions)

            if selected_action == SystemAct.ASK_TYPE:
                params = self._generate_params(selected_action)
            else:
                raise ValueError("disallowed sys_Act {}".format(selected_action))

            return Action(selected_action, params)
        else:
            # if queryable, need to present result
            if len(self.state['informed']['name']) > 0:
                # specific restaurant
                cur_info = {'name': self.state['informed']['name'][-1]}
            else:
                cur_info = {entity: self.state['informed'][entity][-1] for entity in ['pricerange', 'area', 'food'] \
                            if self.state['informed'][entity][-1] != dialog_config.I_DO_NOT_CARE}
            match_result = self.query_in_DB(cur_info, skip=self.state['results'])
            if len(match_result) > 0:
                present_result = match_result[0]
                #self.state['results'].append(present_result)
                #self.state['match_presented'] += 1
                params = present_result#{present_result[entity] for entity in dialog_config.informable_slots}
                return Action(SystemAct.PRESENT_RESULT, params)
            else:
                return Action(SystemAct.NOMATCH_RESULT, None)

    def _response_INFORM_TYPE_CHANGE(self, usr_act):
        # print(usr_act)
        # print(self.state['informed'])
        if len(self.state['informed']['name']) > 0:
            # specific restaurant
            cur_info = {'name': self.state['informed']['name'][-1]}
        else:
            cur_info = {entity: self.state['informed'][entity][-1] for entity in ['pricerange', 'area', 'food'] \
                        if len(self.state['informed'][entity]) and self.state['informed'][entity][-1] != dialog_config.I_DO_NOT_CARE}

        match_result = self.query_in_DB(cur_info, skip=self.state['results'])
        if len(match_result) > 0:
            present_result = match_result[0]
            #self.state['results'].append(present_result)
            #self.state['match_presented'] += 1
            params = present_result#{present_result[entity] for entity in dialog_config.informable_slots}
            return Action(SystemAct.PRESENT_RESULT, params)
        else:
            return Action(SystemAct.NOMATCH_RESULT, None)

    def _response_ASK_INFO(self, usr_act):
        params = self._generate_params(SystemAct.PROVIDE_INFO, usr_act=usr_act)
        return Action(SystemAct.PROVIDE_INFO, params)

    def _response_MAKE_RESERVATION(self, usr_act):

        params = {entity: None for entity, value in self.state['reservation_informed'].items() if len(value) == 0}
        if len(params) == 0:
            # all asked
            if self.sample([True, False], p=[0.8, 0.2]):
                params = {'reference': "ABC"}
                return Action(SystemAct.BOOKING_SUCCESS, params)
            else:
                return Action(SystemAct.BOOKING_FAIL, None)
        else:
            return Action(SystemAct.ASK_RESERVATION_INFO, params)


    def _response_MAKE_RESERVATION_CHANGE_TIME(self, usr_act):
        if self.sample([True, False], p=[0.8, 0.2]):
            params = {'reference': "ABC"}
            return Action(SystemAct.BOOKING_SUCCESS, params)
        else:
            return Action(SystemAct.BOOKING_FAIL, None)

    def _response_ANYTHING_ELSE(self):
        if len(self.state['informed']['name']) > 0:
            # specific restaurant
            cur_info = {'name': self.state['informed']['name'][-1]}
        else:
            cur_info = {entity: self.state['informed'][entity][-1] for entity in ['pricerange', 'area', 'food'] \
                        if len(self.state['informed'][entity]) and self.state['informed'][entity][-1] != dialog_config.I_DO_NOT_CARE}

        match_result = self.query_in_DB(cur_info, skip=self.state['results'])
        if len(match_result) > 0: #self.state['match_presented']:
            present_result = match_result[0]#match_result[self.state['match_presented']]
            params = present_result
            return Action(SystemAct.PRESENT_RESULT, params)
        else:
            return Action(SystemAct.NO_OTHER, None)

    def _response_GOODBYE(self):
        return Action(SystemAct.GOODBYE, None)






if __name__ == "__main__":
    system = System()