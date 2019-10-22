import numpy as np
from simulator.user import User
from simulator import dialog_config
from simulator.agent.core import SystemAct, UserAct, Action


class LooseUser(User):
    def __init__(self, nlg_sample, nlg_template):
        super().__init__(nlg_sample=nlg_sample, nlg_template=nlg_template)

    def check_presented_result(self, match):
        """
        checke the presented_result/no_match_result
        :return:
        """
        if match == dialog_config.NO_MATCH:
            query_result = self.query_in_DB(self.goal['cur_info'], skip=self.state['results'])
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
            # at_least_one_requirements_informed = [(self.state['informed'][entity] > 0) for entity in self.goal['cur_info']]
            # at_least_one_requirements_informed = np.any(at_least_one_requirements_informed)
            # if at_least_one_requirements_informed:
            #     for k, v in self.goal['cur_info'].items():
            #         if v != match[k]:
            #             print("the presented_result doesn't match the requirement!")
            #             return dialog_config.CONSTRAINT_CHECK_FAILURE
            #     return dialog_config.CONSTRAINT_CHECK_SUCCESS
            # else:
                # the user hasn't informed all the slots
            tmp_constraint_check = [(self.goal['cur_info'][entity] == match[entity]) for entity, value in self.state['informed'].items() \
                                    if ((value > 0) and (entity in self.goal['cur_info']))]

            if len(tmp_constraint_check) and np.all(tmp_constraint_check):
                print("Warning, the system hasn't captured all the correct entity but gives the result anyway")
                return dialog_config.CONSTRAINT_CHECK_SUCCESS
            else:
                print("Warning, the system hasn't captured all the correct entity but gives the result anyway, and the result is not correct")
                return dialog_config.CONSTRAINT_CHECK_FAILURE

            return dialog_config.CONSTRAINT_CHECK_FAILURE
            raise ValueError("the user hasn't informed all requirements! but the system presents the result already.")

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
