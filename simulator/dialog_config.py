'''
Created on May 17, 2016

@author: xiul, t-zalipt
'''

# sys_request_slots = ['moviename', 'theater', 'starttime', 'date', 'numberofpeople', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids']
# # sys_inform_slots = ['moviename', 'theater', 'starttime', 'date', 'genre', 'state', 'city', 'zip', 'critic_rating', 'mpaa_rating', 'distanceconstraints', 'video_format', 'theater_chain', 'price', 'actor', 'description', 'other', 'numberofkids', 'taskcomplete', 'ticket']
from simulator.agent.core import SystemAct

informable_slots = ["food", "area", "pricerange", "name"]
requestable_slots =  ["address", "phone", "postcode"]#, "food", "area", "pricerange"]
reservation_slots =  ["people", "time", "day"]

# start_dia_acts = {
#     #'greeting':[],
#     'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
# }
MAX_TURN = 10



################################################################################
# user action_to_index dictionary
################################################################################
from simulator.agent.core import UserAct

USER_ACTION_TO_INDEX = {v: i-2 for i, (a, v) in enumerate(UserAct.__dict__.items()) if not a.startswith("__")}

USER_ACT_CARDINALITY = len(USER_ACTION_TO_INDEX)

action_to_index_dict = {SystemAct.ASK_TYPE: 0,
                        SystemAct.PRESENT_RESULT: 1,
                        SystemAct.NOMATCH_RESULT: 1,
                        SystemAct.NO_OTHER: 1,
                        SystemAct.PROVIDE_INFO: 2,
                        SystemAct.BOOKING_SUCCESS: 3,
                        SystemAct.BOOKING_FAIL: 3,
                        SystemAct.GOODBYE: 4,
                        SystemAct.ASK_RESERVATION_INFO: 5}

index_to_action_dict = {0: SystemAct.ASK_TYPE,
                        1: [SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT, SystemAct.NO_OTHER],
                        2: SystemAct.PROVIDE_INFO,
                        3: [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL],
                        4: SystemAct.GOODBYE,
                        5: SystemAct.ASK_RESERVATION_INFO}

SYS_ACTION_CARDINALITY = max(action_to_index_dict.values()) + 1

STATE_DIM = 33 if len(requestable_slots) == 3 else 39

################################################################################
# training mode
################################################################################
RL_WARM_START = 1
RL_TRAINING = 2
RANDOM_ACT = 3
INTERACTIVE = 4

################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = 'failed_dialog'
SUCCESS_DIALOG = 'success_dialog'
NO_OUTCOME_YET = 'no_outcome_yet'
TURN_SUCCESS_FOR_SL = 'turn_success_for_sl'
TURN_FAIL_FOR_SL = 'turn_fail_for_sl'

# Rewards
SUCCESS_REWARD = 1#50
FAILURE_REWARD = -1#-10
PER_TURN_REWARD = -0.1#-1#
TURN_AWARD = 1
repetition_penalty = -0.5

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "dontcare"# do not change
NO_MATCH = "NO VALUE MATCHES!!!"
NO_OTHER = "NO other restaurants matches!"
AT_MOST_ANYTHING_ELSE = 1

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = -1
CONSTRAINT_CHECK_NOTYET = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  Provided Info Check
################################################################################
INFO_CHECK_FAILURE = -1
INFO_CHECK_NOTYET = 0
INFO_CHECK_SUCCESS = 1

################################################################################
#  Reservation Attempt Check
################################################################################
RESERVATION_CHECK_FAILURE = -1
RESERVATION_CHECK_NOTYET = 0
RESERVATION_CHECK_SUCCESS = 1


################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 10

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    #{'diaact':"greeting", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_question actions
    ############################################################################
    {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   confirm_answer actions
    ############################################################################
    {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   deny actions
    ############################################################################
    {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
]
############################################################################
#   Adding the inform actions
############################################################################
# for slot in sys_inform_slots:
#     feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})
#
# ############################################################################
# #   Adding the request actions
# ############################################################################
# for slot in sys_request_slots:
#     feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})

