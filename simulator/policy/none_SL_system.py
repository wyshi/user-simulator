import sys
sys.path.append("/home/wyshi/simulator")
from simulator.user import User, Goal
from simulator.loose_user import LooseUser
from simulator.loose_system import LooseSystem
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from simulator.agent.core import SystemAct
from rl.policy_model import Net
from rl.my_pg import PolicyGradientREINFORCE
from config import Config

config = Config()
MODE = dialog_config.INTERACTIVE
if MODE == dialog_config.INTERACTIVE:
    config.INTERACTIVE = True
config.with_bit = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM   = dialog_config.STATE_DIM
NUM_ACTIONS = dialog_config.SYS_ACTION_CARDINALITY


def get_bit_vector(system):
    reservable = [len(value) for entity, value in system.state['reservation_informed'].items()]
    reservable = np.all(reservable)
    small_value = config.small_value
    if len(system.state['informed']['name']) > 0:
        bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
        bit_vecs[4] = small_value
        if not reservable:
            bit_vecs[3] = small_value
        else:
            bit_vecs[3] = 1
        if len(system.state['results']) == 0:
            bit_vecs[2] = small_value
            bit_vecs[5] = small_value
        else:
            bit_vecs[2] = 1
            bit_vecs[5] = 1
        return bit_vecs

    informed_so_far = [len(value) > 0 for entity, value in system.state['informed'].items() if entity != 'name']

    assert len(informed_so_far)
    if np.any(informed_so_far):
        bit_vecs = [1] * dialog_config.SYS_ACTION_CARDINALITY
        bit_vecs[4] = small_value
        if not reservable:
            bit_vecs[3] = small_value
        else:
            bit_vecs[3] = 1

        if len(system.state['results']) == 0:
            bit_vecs[2] = small_value
            bit_vecs[5] = small_value
        else:
            bit_vecs[2] = 1
            bit_vecs[5] = 1

        return bit_vecs
    else:
        bit_vecs = [1, small_value, small_value, small_value, small_value, small_value]
        return bit_vecs


def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    model = torch.load(model_dir)
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY).to(device)
    net.load_state_dict(model)
    net.eval()
    return net


policy_net = load_policy_model("model/save/no_bit_no_warm_up/test_nlg_no_warm_up_with_nlu_0_2019-5-11-15-54-44-5-131-1.pkl")


optimizer = optim.RMSprop(policy_net.parameters())

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=STATE_DIM,
                     num_actions=NUM_ACTIONS,
                     device=device,
                     init_exp=config.init_exp,         # initial exploration prob
                     final_exp=config.final_exp,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.9, # discount future rewards
                     reg_param=0.1,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
                     batch_size=config.batch_size,
                     verbose=True,
                     with_bit=config.with_bit,
                     replay=config.replay)

user = User(nlg_sample=False)
system = LooseSystem()
env = Enviroment(user=user, system=system, verbose=True, config=config)
sys_act = None
status = []

while True:
    print("-"*20)
    # turker_response =
    state = env.reset(mode=MODE) # turker_response
    sys_act = None # initial sys act
    total_rewards = 0
    while True:
        # print(state)
        # print(env.system.state)
        if config.with_bit:
            bit_vecs = get_bit_vector(system)
        else:
            bit_vecs = None

        action = pg_reinforce.sampleAction(state[np.newaxis, :], bit_vecs=bit_vecs,
                                                   rl_test=True)
        provided_sys_act = action.item()
        # print(provided_sys_act)
        result_step_sys = env.step_system(provided_sys_act=provided_sys_act, mode=MODE)

        if result_step_sys is not None:
            # goes into FAILED_DIALOG, shouldn't happen in rule_policy and INTERACTIVE mode
            next_state, reward, env.done = result_step_sys
            sys_sent = "Sorry, an error message occurred."#env.last_sys_sent
            print(sys_sent)
        else:
            sys_sent = env.last_sys_sent
            print(sys_sent)

            # turker_response =
            next_state, reward, env.done = env.step_user(mode=MODE) # turker_response

        total_rewards += reward
        pg_reinforce.storeRollout(state, action, reward, bit_vecs=bit_vecs)

        state = next_state

        if env.done:
            sys_sent = env.last_sys_sent
            print(sys_sent)
            status.append(env.success)
            # assert env.success
            print('dialog_status: {}'.format(env.success))
            print('reward: {}'.format(total_rewards))
            print("-"*20)
            break


