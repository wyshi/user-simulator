import sys
sys.path.append("/home/wyshi/simulator")
from simulator.loose_user import LooseUser
# from simulator.user import Goal
from simulator.user import User
from simulator.system import System
from simulator.loose_system import LooseSystem
from sequicity_user.seq_user import Seq_User
from sequicity_user.seq_user_act import Seq_User_Act
from simulator.env_for_evaluation import Enviroment
import simulator.dialog_config as dialog_config
import numpy as np
from simulator.agent.core import SystemAct
from config import Config
from evaluation.config import Config as evaluation_config

from tqdm import tqdm
eval_config = evaluation_config()
config = Config()

if eval_config.rule_policy:
    if eval_config.nlg_template:
        user = LooseUser(nlg_sample=False)
    elif eval_config.nlg_sample:
        user = LooseUser(nlg_sample=True)
    elif eval_config.nlg_generation:
        pass
else:
    if eval_config.nlg_template:
        user = Seq_User_Act(nlg_sample=False)
    elif eval_config.nlg_sample:
        user = Seq_User_Act(nlg_sample=True)
    elif eval_config.nlg_generation:
        user = Seq_User()

system = System(config=config) # sequicity system
env = Enviroment(user=user, system=system, verbose=True, config=config)

sys_act = None
status = []
MODE = dialog_config.RL_WARM_START#RANDOM_ACT#RL_WARM_START#RANDOM_ACT#RL_WARM_START#INTERACTIVE#RL_TRAINING#RANDOM_ACT#RL_WARM_START

for _ in tqdm(range(100)):
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
        provided_sys_act = None
        next_state, reward, done = env.step(provided_sys_act=provided_sys_act, mode=MODE)
        print("env.last_usr_act_true", env.last_usr_act_true)
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
            print("-"*20)
            print("\n\n\n")
            break


