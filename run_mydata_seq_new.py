from collections import deque

from rl.my_pg_sequicity import PolicyGradientREINFORCE
from rl.policy_model import Net
from beeprint import pp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
# import gym

from sequicity.model import Model

# env_name = 'CartPole-v0'
# env = gym.make(env_name)

from simulator.user import User
from simulator.loose_user import LooseUser
from simulator.system import System
from simulator.loose_system import LooseSystem
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config
import numpy as np
from config import Config
from simulator.agent.core import SystemAct, UserAct
from sequicity.config import global_config as seq_cfg
from sequicity.model import load_rl_model
import tensorflow as tf

tf.app.flags.DEFINE_bool("one_hot", True, "The path to save the model")
tf.app.flags.DEFINE_bool("new_reward", True, "The path to the csv")
tf.app.flags.DEFINE_bool("nlg_sample", True, "The path to the csv")
tf.app.flags.DEFINE_integer("with_bit", None, "The path to the csv")
tf.app.flags.DEFINE_string("save_dir", None, "The path to the csv")
args = tf.app.flags.FLAGS

dummy = args.one_hot # hack to get all the configs
# pdb.set_trace()
config = Config()
device = config.device#torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.one_hot:
    config.use_sent_one_hot = True
    config.use_sent = False
else:
    config.use_sent_one_hot = False
    config.use_sent = True

if args.new_reward:
    config.use_new_reward = True
else:
    config.use_new_reward=False

if args.with_bit == 0:
    config.with_bit = False
    config.with_bit_rep_only = False
    with_bit_more = False
    with_bit_all = False
elif args.with_bit == 1:
    config.with_bit = True
    config.with_bit_rep_only = True
    with_bit_more = False
    with_bit_all = False
elif args.with_bit == 2:
    config.with_bit = True
    config.with_bit_rep_only = False
    with_bit_more = True
    with_bit_all = False
elif args.with_bit == 3:
    config.with_bit = True
    config.with_bit_rep_only = False
    with_bit_more = False
    with_bit_all = True

if args.nlg_sample:
    config.nlg_sample = True
else:
    config.nlg_sample = False

if args.save_dir:
    config.save_dir = args.save_dir




if config.loose_agents:
    user = LooseUser(nlg_sample=False)
    system = LooseSystem()
else:
    user = User(nlg_sample=False)
    system = System()

env = Enviroment(user=user, system=system, verbose=True)
sys_act = None
status = []

state_dim   = dialog_config.STATE_DIM
num_actions = dialog_config.SYS_ACTION_CARDINALITY


def run_one_dialog(env, pg_reinforce):
    print("Test Episode "+"-"*20)
    cur_mode = dialog_config.RL_TRAINING

    usr_act_seq = []
    print("-" * 20)
    # initialize
    state = env.reset(mode=MODE)
    usr_act_seq.append(env.last_usr_act_true)
    print("*" * 20)
    z_input = accum_slots(usr_act_seq)
    print(z_input)
    print("*" * 20)
    state = [env.last_usr_sent, 0, None, None, {}, z_input] + [state]

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

        print(state)
        action, turn_num, prev_z, turn_states = pg_reinforce.sampleAction(state,
                                                                          bit_vecs=bit_vecs,
                                                                          rl_test=True)
        action = action.item()
        next_state, reward, done = env.step(provided_sys_act=action, mode=cur_mode)
        usr_act_seq.append(env.last_usr_act_true)
        print("*" * 20)
        z_input = accum_slots(usr_act_seq)
        print(z_input)
        print("*" * 20)
        next_state = [env.last_usr_sent, turn_num, env.last_sys_sent, prev_z, turn_states, z_input] + [next_state]

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

    return total_rewards, total_t, env.success


def test(env, pg_reinforce, n=50):
    reward_list = []
    dialogLen_list = []
    success_list = []
    # print(i_episode)
    for i_test in range(n):
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

# policy_net = Net(state_dim=state_dim, num_actions=num_actions).to(device)#
policy_net = load_rl_model(discrete_act=config.discrete_act, pretrained_dir=None)
# turn_num, m_idx, prev_z, turn_states, pz_proba = policy_net.rl_interactive_single_turn('i want a chinese restaurant', 0)

optimizer = optim.Adam(lr=config.lr, params=filter(lambda x: x.requires_grad, policy_net.m.parameters()),
                                  weight_decay=5e-5)



pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=state_dim,
                     num_actions=num_actions  ,
                     discrete_act=config.discrete_act,
                     device=device,
                     init_exp=config.init_exp,         # initial exploration prob
                     final_exp=config.final_exp,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=config.discounted_factor, # discount future rewards
                     reg_param=0.1,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
    batch_size=config.batch_size,
    verbose=True,
    with_bit=config.with_bit,
    replay=config.replay)

WARM_START_EPISODES = config.warm_start_episodes
MAX_EPISODES = config.n_episodes
MAX_STEPS    = 200
TEST_EVERY = 1000
NUM_TEST = 100
MODE = dialog_config.RL_WARM_START
import time

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
            mean_reward_test.append(np.mean(reward_list))
            mean_len_test.append(np.mean(len_list))
            mean_success_test.append(np.mean(success_list))
            test_id.append(i_episode - WARM_START_EPISODES)

        usr_act_seq = []
        print("-" * 20)
        # initialize
        state = env.reset(mode=MODE)
        usr_act_seq.append(env.last_usr_act_true)
        print("*" * 20)
        z_input = accum_slots(usr_act_seq)
        print(z_input)
        print("*" * 20)


        state = [env.last_usr_sent, 0, None, None, {}, z_input] + [state]
        total_rewards = 0
        total_t = 0

        while True:
            # env.render()
            if config.with_bit:
                bit_vecs = get_bit_vector(system)
            else:
                bit_vecs = None
            #print('bit_vec: ', bit_vecs)
            if MODE == dialog_config.RL_TRAINING:
                action, turn_num, prev_z, turn_states = pg_reinforce.sampleAction(state,
                                                                                  bit_vecs=bit_vecs)
                if config.discrete_act:
                    action = action.item()
            elif MODE == dialog_config.RL_WARM_START:
                action = None
            if not config.discrete_act:
                sent, filled_sent, sent_parameters = policy_net.fill_sentence(action, prev_z)

            next_state, reward, done = env.step(provided_sys_act=action, mode=MODE)
            usr_act_seq.append(env.last_usr_act_true)
            print("*" * 20)
            z_input = accum_slots(usr_act_seq)
            print(z_input)
            print("*" * 20)
            next_state = [env.last_usr_sent, turn_num, env.last_sys_sent, prev_z, turn_states, z_input]+[next_state]


            total_rewards += reward

            if MODE == dialog_config.RL_WARM_START:
                # print("here")
                action = env.system.action_to_index(env.last_sys_act.act)
            # print(action)
            pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs)

            state = next_state
            total_t += 1
            if done:
                break

        pg_reinforce.updateModel(mode=MODE)

        episode_history.append(total_rewards)
        mean_rewards = np.mean(episode_history)

        print("Episode {}".format(i_episode))
        print("Finished after {} timesteps".format(total_t+1))
        print('dialog_status: {}'.format(env.success))
        print("Reward for this episode: {}".format(total_rewards))
        print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
        print("\n\n\n")
        if mean_rewards >= 48.0 and len(episode_history) >= 100:
            print("Environment {} solved after {} episodes".format("Restaurant", i_episode+1))
            break

        if i_episode > 2500 and mean_rewards < -9:
            break

        if MODE == dialog_config.RL_TRAINING and \
           (((i_episode - WARM_START_EPISODES + 1) % TEST_EVERY == 0)): #or (i_episode == WARM_START_EPISODES)):

            print(mean_reward_test)
            test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)

            pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(config.save_dir + str(cnt) + "_" + cur_time + ".csv", index=False)
            # pg_reinforce.saver.save(sess, SAVE_DIR, write_meta_graph=False)
            if mean_success_test[-1] >= MAX_TEST_SUC:
                MAX_TEST_SUC = mean_success_test[-1]
                torch.save(policy_net.m.state_dict(), config.save_dir  + str(cnt) + "_" + cur_time + ".pkl")

    if mean_success_test[-1] >= MAX_TEST_SUC:
        MAX_TEST_SUC = mean_success_test[-1]
        print("max_test_success in the end", MAX_TEST_SUC)
        torch.save(policy_net.m.state_dict(), config.save_dir  + str(cnt) + "_" + cur_time + ".pkl")

    print(mean_reward_test)
    test_history = zip(test_id, mean_reward_test, mean_len_test, mean_success_test)

    pd.DataFrame(test_history, columns=["id", "reward", "len", "success"]).to_csv(
        config.save_dir  + str(cnt) + "_" + cur_time + ".csv", index=False)

    if i_episode == (MAX_EPISODES-1):
        break

    cnt += 1

