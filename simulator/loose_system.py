import numpy as np
from simulator.system import System
from simulator import dialog_config
from simulator.agent.core import SystemAct, UserAct, Action

class LooseSystem(System):
    def __init__(self, config):
        super().__init__(config=config)

    def _index_to_action(self, sys_act_idx, usr_act=None):
        assert isinstance(sys_act_idx, (int, np.integer))
        if sys_act_idx == 1:
            # present result
            queryable = ((len(self.state['informed']['pricerange']) > 0) or (
                        len(self.state['informed']['area']) > 0) or (len(self.state['informed']['food']) > 0)) \
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
            cur_info = {}
            for entity in ['pricerange', 'area', 'food']:
                if len(self.state['informed'][entity]) > 0:
                    if self.state['informed'][entity][-1] != dialog_config.I_DO_NOT_CARE:
                        cur_info[entity] = self.state['informed'][entity][-1]

        match_result = self.query_in_DB(cur_info, skip=self.state['results'])
        if len(match_result) > 0:
            present_result = match_result[0]
            params = present_result  # {present_result[entity] for entity in dialog_config.informable_slots}
            return Action(SystemAct.PRESENT_RESULT, params)
        else:
            if UserAct.ANYTHING_ELSE in self.state['usr_act_sequence'] and len(match_result) == 0:
                return Action(SystemAct.NO_OTHER, None)
            else:
                return Action(SystemAct.NOMATCH_RESULT, None)
