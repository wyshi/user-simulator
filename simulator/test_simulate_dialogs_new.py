import sys
sys.path.append("/home/wyshi/simulator")
from simulator.loose_user import LooseUser
# from simulator.user import Goal
from simulator.user import User
from simulator.system import System
from simulator.loose_system import LooseSystem
from sequicity_user.seq_user import Seq_User
from sequicity_user.seq_user_act import Seq_User_Act
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config
import numpy as np
from simulator.agent.core import SystemAct
from config import Config
config = Config()

config.use_new_reward = False
def accum_slots(usr_act_turns):
    inform_hist = {}
    book_inform_hist = {}
    output_str = []

    for usr_act in usr_act_turns:

        if usr_act.act in ['inform_type', 'inform_type_change']:
            inform_hist.update(usr_act.parameters)

        elif usr_act.act in ['make_reservation', 'make_reservation_change_time']:
            book_inform_hist.update(usr_act.parameters)

    for slot_name in inform_hist.keys():
        output_str.append(inform_hist[slot_name])
    output_str.append('EOS_Z1')

    for slot_name in book_inform_hist.keys():
        output_str.append(book_inform_hist[slot_name])
    output_str.append('EOS_Z3')

    if usr_act_turns[-1].act in ['request']:
        for slot in usr_act_turns[-1].parameters:
            output_str.append(slot)
    output_str.append('EOS_Z2')

    return ' '.join(output_str)

TEST_SEQ_USER = False

# if False:
#     user = LooseUser(nlg_sample=False)
#     system = LooseSystem(config=config)
# else:
#     user = User(nlg_sample=False, nlg_template=False)
#     system = System(config=config)

user = User(nlg_sample=False, nlg_template=False)
system = System(config=config)


if TEST_SEQ_USER:
    if config.use_sl_generative:
        user = Seq_User(nlg_sample=False, nlg_template=False)
    else:
        user = Seq_User_Act(nlg_sample=True, nlg_template=False)





env = Enviroment(user=user, system=system, verbose=True, config=config)
sys_act = None
status = []
MODE = dialog_config.RL_WARM_START#RANDOM_ACT#RL_WARM_START#RANDOM_ACT#RL_WARM_START#INTERACTIVE#RL_TRAINING#RANDOM_ACT#RL_WARM_START

mean_reward_test = []
mean_len_test = []
mean_success_test = []

for _ in tqdm(range(200)):
    print("-"*20)
    usr_act_seq = []
    next_state = env.reset(mode=MODE)
    usr_act_seq.append(env.last_usr_act_true)
    # print("*"*20)
    # print(accum_slots(usr_act_seq))
    # print("*"*20)
    sys_act = None # initial sys act
    total_rewards = 0
    while True:
        if MODE == dialog_config.RANDOM_ACT:
            provided_sys_act = np.random.choice(range(6))
            index_to_action_dict = {0: SystemAct.ASK_TYPE,
                                    1: [SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT, SystemAct.NO_OTHER],
                                    2: SystemAct.PROVIDE_INFO,
                                    3: [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL],
                                    4: SystemAct.GOODBYE,
                                    5: SystemAct.ASK_RESERVATION_INFO}
            print(index_to_action_dict[provided_sys_act])

        else:
            provided_sys_act = None
        next_state, reward, done = env.step(provided_sys_act=provided_sys_act, mode=MODE)
        print(env.last_usr_act_true)
        usr_act_seq.append(env.last_usr_act_true)
        # print("*" * 20)
        # print(accum_slots(usr_act_seq))
        # print("per turn reward", reward)
        # print("*" * 20)

        total_rewards += reward
        # usr_act, usr_sent = user.respond(sys_act=sys_act)
        # sys_act, sys_sent = system.respond(usr_sent=usr_sent, warm_start=True, usr_act=usr_act)
        # sys_act = next_sys_act
        print("user turn status: ", env.user.dialog_status)
        if done:
            status.append(user.dialog_status)
            # assert env.success
            print('dialog_status: {}'.format(env.success))
            print('reward: {}'.format(total_rewards))
            mean_reward_test.append(total_rewards)
            mean_success_test.append(env.success)
            mean_len_test.append(env.step_i)
            print("-"*20)
            print("\n\n\n")
            break

import pandas as pd
import time
cnt = 'rule-base-system'
cur_time = "-".join([str(t) for t in list(time.localtime())])
full_result = zip(mean_reward_test, mean_len_test, mean_success_test)
pd.DataFrame(full_result, columns=["reward", "len", "success"]).to_csv(
    "evaluation/cross_test/1_rule-base-system/" + str(cnt) + "_" + cur_time + "_full.csv", index=False)
