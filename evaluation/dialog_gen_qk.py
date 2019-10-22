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
from evaluation.eval_config import Config as evaluation_config
from nltk import word_tokenize
import torch
import random

from tqdm import tqdm
eval_config = evaluation_config()
config = Config()

torch.manual_seed(2)
torch.cuda.manual_seed(2)
random.seed(2)
np.random.seed(2)

if eval_config.rule_policy:
    if eval_config.nlg_template:
        user = LooseUser(nlg_sample=False, nlg_template=True)
    elif eval_config.nlg_sample:
        user = LooseUser(nlg_sample=True, nlg_template=False)
    elif eval_config.nlg_generation:
        user = LooseUser(nlg_sample=False, nlg_template=False)
else:
    if eval_config.nlg_template:
        user = Seq_User_Act(nlg_sample=False, nlg_template=True)
    elif eval_config.nlg_sample:
        user = Seq_User_Act(nlg_sample=True, nlg_template=False)
    elif eval_config.nlg_generation:
        user = Seq_User(nlg_sample=False, nlg_template=False)

system = System(config=config) # sequicity system
env = Enviroment(user=user, system=system, verbose=True, config=config)



sys_act = None
status = []
MODE = dialog_config.RL_WARM_START#RANDOM_ACT#RL_WARM_START#RANDOM_ACT#RL_WARM_START#INTERACTIVE#RL_TRAINING#RANDOM_ACT#RL_WARM_START
dial_id_list = []
utterance_list = []
speaker_list = []
dial_act_list = []


for dial_id in tqdm(range(200)):
    print("-"*20)
    usr_act_seq = []
    next_state = env.reset(mode=MODE)

    ##############generate corpus##################
    # # goal
    utterance_list.append(user.goal)
    dial_id_list.append('dial_'+str(dial_id))
    speaker_list.append('goal')
    dial_act_list.append(None)
    # # user utterance
    utterance_list.append(env.last_usr_sent)
    dial_id_list.append('dial_'+str(dial_id))
    speaker_list.append('usr')
    dial_act_list.append(env.last_usr_act_true.act)
    ##############generate corpus##################

    usr_act_seq.append(env.last_usr_act_true)
    # print("*"*20)
    # print(accum_slots(usr_act_seq))
    # print("*"*20)
    sys_act = None # initial sys act
    total_rewards = 0
    while True:
        provided_sys_act = None
        print(user.state['informed'])
        next_state, reward, done = env.step(provided_sys_act=provided_sys_act, mode=MODE)

        ##############generate corpus##################
        # # systen response
        utterance_list.append(env.last_sys_sent)
        dial_id_list.append('dial_'+str(dial_id))
        speaker_list.append('sys')
        dial_act_list.append(env.last_sys_act.act)
        # # user utterance
        utterance_list.append(env.last_usr_sent)
        dial_id_list.append('dial_'+str(dial_id))
        speaker_list.append('usr')
        dial_act_list.append(env.last_usr_act_true.act)
        ##############generate corpus##################

        # print("env.last_usr_act_true", env.last_usr_act_true)
        usr_act_seq.append(env.last_usr_act_true)
        # print("*" * 20)
        # print(accum_slots(usr_act_seq))
        # print("per turn reward", reward)
        # print("*" * 20)

        total_rewards += reward
        # usr_act, usr_sent = user.respond(sys_act=sys_act)
        # sys_act, sys_sent = system.respond(usr_sent=usr_sent, warm_start=True, usr_act=usr_act)
        # sys_act = next_sys_act
        # print("user turn status: ", env.user.dialog_status)
        if done:
            status.append(user.dialog_status)
            # # assert env.success
            # print('dialog_status: {}'.format(env.success))
            # print('reward: {}'.format(total_rewards))
            # print("-"*20)
            # print("\n\n\n")
            break

# df = pd.DataFrame(zip(dialog_ids, whos, dialogs))

# df.to_csv("data/multiwoz-master/data/multi-woz/restaurant.csv", encoding='utf-8')

if eval_config.rule_policy:
    filename = 'rule'
else:
    filename = 'seq'

if eval_config.nlg_template:
    filename += '_template'
elif eval_config.nlg_sample:
    filename += '_sample'
else:
    filename += '_generation'

import pandas as pd


df = pd.DataFrame(zip(dial_id_list, speaker_list, utterance_list, dial_act_list), columns= ['dial_id', 'speaker','utterance','dialog_act'])
df.to_csv('./evaluation/dial_log/' + filename + '.csv', encoding='utf-8', index = None, header=False)



df = pd.DataFrame(zip(dial_id_list, speaker_list, utterance_list), columns= ['dial_id', 'speaker','utterance'])
df.to_csv('./evaluation/hu_eval_dial_log/' + filename + '_hu_eval.csv', encoding='utf-8', index = None, header=True)











