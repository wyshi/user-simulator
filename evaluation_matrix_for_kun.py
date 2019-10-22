from collections import deque

from rl.my_pg import PolicyGradientREINFORCE
from rl.policy_model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
# import gym
from beeprint import pp
from simulator.agent.core import Action, UserAct, SystemAct
# from sequicity.model import Model

# env_name = 'CartPole-v0'
# env = gym.make(env_name)

from simulator.user import User
from simulator.loose_user import LooseUser
from sequicity_user.seq_user import Seq_User
from sequicity_user.seq_user_act import Seq_User_Act
from simulator.system import System
from simulator.loose_system import LooseSystem
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config
import numpy as np
from evaluation_matrix_config import EvalConfig
import pdb
from simulator.agent.core import SystemAct
from sequicity.config import global_config as seq_cfg
import argparse

from tqdm import tqdm
# from sequicity.model import load_rl_model

config = EvalConfig()
device = config.device

# ########################## #########################
parser = argparse.ArgumentParser()
parser.add_argument('-resume_rl_model_dir')
parser.add_argument('-config', nargs='*')
args = parser.parse_args()

if args.config:
    for pair in args.config:
        k, v = tuple(pair.split('='))
        dtype = type(getattr(config, k))
        if dtype == type(None):
            raise ValueError()
        if dtype is bool:
            v = False if v == 'False' else True
        else:
            v = dtype(v)
        setattr(config, k, v)
# print(config.use_sl_simulator)
# print(config.nlg_sample)
# print(config.nlg_template)
# print(args.resume_rl_model_dir)

if config.use_sl_simulator == True:
    config.use_new_reward = False
else:
    config.use_new_reward = True

# if config.nlg_template == True:
#     config.nlg_sample = False
# else:
#     config.nlg_sample = True
# pdb.set_trace()
# exit()
# # ########################## #########################

if config.use_sl_simulator:
    username = 'seq'
else:
    username = 'rule'

if config.use_sl_generative:
    username += '_generation'
elif config.nlg_template:
    username += '_template'
else:
    username += '_sample'
    

if args.resume_rl_model_dir.split('/')[6] == 'sl_simulator':
    sysname = 'sl_' + args.resume_rl_model_dir.split('/')[7]
else:
    sysname = 'rule_' + args.resume_rl_model_dir.split('/')[6]

output_name = username + '_' + sysname


if config.loose_agents:
    user = LooseUser(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = LooseSystem(config=config)
else:
    user = User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    system = System(config=config)

if config.use_sl_simulator:
    if config.use_sl_generative:
        user = Seq_User(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)
    else:
        user = Seq_User_Act(nlg_sample=config.nlg_sample, nlg_template=config.nlg_template)


pp(config)
pp(dialog_config)



env = Enviroment(user=user, system=system, verbose=True, config=config)
sys_act = None
status = []

state_dim   = dialog_config.STATE_DIM
num_actions = dialog_config.SYS_ACTION_CARDINALITY


def run_one_dialog(env, pg_reinforce):
    print("#"*30)
    print("Test Episode "+"-"*20)
    print("#"*30)
    cur_mode = dialog_config.RL_TRAINING
    state = env.reset(mode=cur_mode)
    total_rewards = 0
    total_t = 0

    while True:
        # env.render()
        # print(state[np.newaxis, :])
        if config.with_bit:
            bit_vecs = get_bit_vector(system)
        else:
            bit_vecs = None
        print('bit_vec: ', bit_vecs)
        action = pg_reinforce.sampleAction(state, rl_test=True, bit_vecs=bit_vecs)
        action = action.item()
        next_state, reward, done = env.step(provided_sys_act=action, mode=cur_mode)

        total_rewards += reward
        # reward = -10 if done else 0.1 # normalize reward
        pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs)

        state = next_state
        total_t += 1
        if done:
            break

    pg_reinforce.cleanUp()
    print("Finished after {} timesteps".format(total_t))
    print('dialog_status: {}'.format(env.success))
    print("Reward for this episode: {}".format(total_rewards))
    print("#" * 30)

    return total_rewards, total_t, env.success

def test(env, pg_reinforce, n=50):
    reward_list = []
    dialogLen_list = []
    success_list = []
    # print(i_episode)
    for i_test in tqdm(range(n)):
        assert len(pg_reinforce.reward_buffer) == 0
        cur_reward, cur_dialogLen, cur_success = run_one_dialog(env, pg_reinforce)
        assert cur_success is not None
        reward_list.append(cur_reward)
        dialogLen_list.append(cur_dialogLen)
        # print(cur_reward)
        # print(cur_dialogLen)
        # print(cur_success)
        success_list.append(int(cur_success))
    return reward_list, dialogLen_list, success_list

def get_bit_vector(system):
    # index_to_action_dict = {0: SystemAct.ASK_TYPE,
    #                         1: [SystemAct.PRESENT_RESULT, SystemAct.NOMATCH_RESULT, SystemAct.NO_OTHER],
    #                         2: SystemAct.PROVIDE_INFO,
    #                         3: [SystemAct.BOOKING_SUCCESS, SystemAct.BOOKING_FAIL],
    #                         4: SystemAct.GOODBYE,
    #                         5: SystemAct.ASK_RESERVATION_INFO}

    if config.with_bit_all:
        # not reservation, 5 is 0; len(results) == 0, 235 are zero; len(informed)==0, 12345 are zero
        # no repetition, if len(informed)==3, 0 is zero; if reservable, 5 is zero
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value
            bit_vecs[0] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1


            if not reservable:
                bit_vecs[3] = small_value
            else:
                bit_vecs[3] = 1
                bit_vecs[5] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        assert len(informed_so_far)
        if np.sum(informed_so_far) > 1:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1

            if not reservable:
                bit_vecs[3] = small_value
            else:
                #bit_vecs[0] = 0
                bit_vecs[3] = 1
                bit_vecs[5] = small_value

            if np.all(informed_so_far):
                bit_vecs[0] = 0

            return bit_vecs
        else:
            bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
            return bit_vecs

    elif config.with_bit_more:
        # not reservation, 5 is 0; len(results) == 0, 235 are zero; len(informed)==0, 12345 are zero
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value
            # bit_vecs[0] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1


            if not reservable:
                bit_vecs[3] = small_value
            else:
                bit_vecs[3] = 1
                # bit_vecs[5] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        assert len(informed_so_far)
        if np.sum(informed_so_far) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[3] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[3] = 1
                bit_vecs[5] = 1

            if not reservable:
                bit_vecs[3] = small_value
            else:
                #bit_vecs[0] = 0
                bit_vecs[3] = 1
                # bit_vecs[5] = small_value

            # if np.all(informed_so_far):
            #     bit_vecs[0] = 0

            return bit_vecs
        else:
            bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
            return bit_vecs

    elif config.with_bit_rep_only:
        reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
        reservable = np.all(reservable)
        small_value = config.small_value
        if len(system.state['informed']['name']) > 0:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            # bit_vecs[4] = small_value
            bit_vecs[0] = small_value

            # if len(system.state['results']) == 0:
            #     bit_vecs[2] = small_value
            #     bit_vecs[3] = small_value
            #     bit_vecs[5] = small_value
            # else:
            #     bit_vecs[2] = 1
            #     bit_vecs[3] = 1
            #     bit_vecs[5] = 1
            #
            # if not reservable:
            #     bit_vecs[3] = small_value
            # else:
            #     bit_vecs[3] = 1
            #     bit_vecs[5] = small_value
            return bit_vecs

        informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

        if np.all(informed_so_far):
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[0] = small_value
        else:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY

        return bit_vecs
        assert len(informed_so_far)
        if np.sum(informed_so_far) > 1:
            bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
            bit_vecs[4] = small_value

            if len(system.state['results']) == 0:
                bit_vecs[2] = small_value
                bit_vecs[5] = small_value
            else:
                bit_vecs[2] = 1
                bit_vecs[5] = 1

            if not reservable:
                bit_vecs[3] = small_value
            else:
                # bit_vecs[0] = 0
                bit_vecs[3] = 1
                bit_vecs[5] = small_value

            if np.all(informed_so_far):
                bit_vecs[0] = 0

            return bit_vecs
        else:
            bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
            return bit_vecs

def accum_slots(usr_act_turns):
    inform_hist = {}
    book_inform_hist = {}
    output_str = []

    for usr_act in usr_act_turns:

        if usr_act.act in [UserAct.INFORM_TYPE, UserAct.INFORM_TYPE_CHANGE]:
            inform_hist.update({k: v for k, v in usr_act.parameters.items() if v != dialog_config.I_DO_NOT_CARE})

        elif usr_act.act in [UserAct.MAKE_RESERVATION, UserAct.MAKE_RESERVATION_CHANGE_TIME]:
            book_inform_hist.update(usr_act.parameters)

    for slot_name in inform_hist.keys():
        output_str.append(inform_hist[slot_name])
    output_str.append('EOS_Z1')

    for slot_name in book_inform_hist.keys():
        output_str.append(book_inform_hist[slot_name])
    output_str.append('EOS_Z3')

    if usr_act_turns[-1].act in [UserAct.ASK_INFO]:
        for slot in usr_act_turns[-1].parameters:
            output_str.append(slot)
    output_str.append('EOS_Z2')

    return ' '.join(output_str)

def fill_sentence(self, m_idx, z_idx):

    sent = self.reader.vocab.sentence_decode(m_idx[0], eos='EOS_M').split()
    slots = [self.reader.vocab.decode(z) for z in z_idx[0]]
    constraints = slots[:slots.index('EOS_Z1')]
    db_results = self.reader.db_search(constraints)

    filled_sent = []
    filled_slot = {}
    import random
    if db_results:
        rand_result = random.choice(db_results)
        for idx, word in enumerate(sent):
            if '_SLOT' in word:
                filled_sent.append(rand_result[word.split('_')[0]])
                filled_slot[word.split('_')[0]] = rand_result[word.split('_')[0]]
            else:
                filled_sent.append(word)

    # filled_sent = ' '.join(sent)
    return " ".join(sent), ' '.join(filled_sent), filled_slot

def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    model = torch.load(model_dir)
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY, config=config).to(device)
    net.load_state_dict(model)
    net.eval()
    return net


policy_net = load_policy_model(args.resume_rl_model_dir)



# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(lr=config.lr, params=policy_net.parameters(),
                                  weight_decay=5e-5)
# net.optimizer = optim.Adam(params=net.parameters(), lr=5e-4, weight_decay=1e-3)
# net.lr_scheduler = optim.lr_scheduler.StepLR(net.optimizer, step_size=500, gamma=0.95)
# net.loss_func = nn.CrossEntropyLoss()

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=state_dim,
                     num_actions=num_actions ,
                     config=config,
                     device=device,
                     init_exp=config.init_exp,         # initial exploration prob
                     final_exp=config.final_exp,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=config.discounted_factor, # discount future rewards
                     reg_param=0.01,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
                     batch_size=config.batch_size,
                     verbose=True,
                     with_bit=config.with_bit,
                     replay=config.replay)


WARM_START_EPISODES = 0#config.warm_start_episodes
MAX_EPISODES = 1
MAX_STEPS    = 200
TEST_EVERY = 1
NUM_TEST = 200
MODE = dialog_config.RL_TRAINING#dialog_config.RL_WARM_START
import time
import random

torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)



# # # ########################## #########################
# print(config.use_sl_simulator)
# print(config.use_sl_generative, config.nlg_template, config.nlg_sample)
# print(args.resume_rl_model_dir)
# print(output_name)
# print(config.use_new_reward)

# pdb.set_trace()

# # # ########################## #########################


MAX_TEST_SUC = -1
cnt = 0
while True:
    print("-------------------START OVER-------------------")
    episode_history = deque(maxlen=100)
    mean_reward_test = []
    mean_len_test = []
    mean_success_test = []
    test_id = []
    cur_time = "-".join([str(t) for t in list(time.localtime())])
    for i_episode in tqdm(range(MAX_EPISODES)):
        if i_episode >= WARM_START_EPISODES:
            MODE = dialog_config.RL_TRAINING

        if MODE == dialog_config.RL_TRAINING and \
           (((i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0)):# or (i_episode == WARM_START_EPISODES)):
            reward_list, len_list, success_list = test(env=env, pg_reinforce=pg_reinforce, n=NUM_TEST)
            full_result = zip(reward_list, len_list, success_list)
            # pd.DataFrame(full_result, columns=["reward", "len", "success"]).to_csv(
            #     config.save_dir + str(cnt) + "_" + cur_time + "_full.csv", index=False)
            pd.DataFrame(full_result, columns=["reward", "len", "success"]).to_csv(
                config.save_dir + output_name + "_full.csv", index=False)
            mean_reward_test.append(np.mean(reward_list))
            mean_len_test.append(np.mean(len_list))
            mean_success_test.append(np.mean(success_list))
            test_id.append(i_episode - WARM_START_EPISODES)

        print("-" * 20)
        # initialize
        state = env.reset(mode=MODE)
        # state = [state, env.last_usr_sent]
        total_rewards = 0
        total_t = 0

    # print(mean_reward_test)
    test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)

    # pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(
    #     config.save_dir + str(cnt) + "_" + cur_time + ".csv", index=False)
    pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(
        config.save_dir + output_name + ".csv", index=False)

    if i_episode == (MAX_EPISODES-1):
        break

    cnt += 1

