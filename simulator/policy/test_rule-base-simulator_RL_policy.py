import sys
sys.path.append("/home/wyshi/simulator")
from simulator.user import User, Goal
from simulator.system import System
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STATE_DIM   = dialog_config.STATE_DIM
NUM_ACTIONS = dialog_config.SYS_ACTION_CARDINALITY


def load_policy_model(model_dir="model/test_nlg_no_warm_up_with_nlu.pkl"):
    model = torch.load(model_dir)
    net = Net(state_dim=dialog_config.STATE_DIM, num_actions=dialog_config.SYS_ACTION_CARDINALITY).to(device)
    net.load_state_dict(model)
    net.eval()
    return net


policy_net = load_policy_model("model/test_nlg_no_warm_up_with_nlu.pkl")


optimizer = optim.RMSprop(policy_net.parameters())

pg_reinforce = PolicyGradientREINFORCE(
                     optimizer=optimizer,
                     policy_network=policy_net,
                     state_dim=STATE_DIM,
                     num_actions=NUM_ACTIONS  ,
                     device=device,
                     init_exp=0.7,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
                     verbose=True)

user = User(nlg_sample=False)
system = System()
env = Enviroment(user=user, system=system, verbose=True)
sys_act = None
status = []
MODE = dialog_config.INTERACTIVE
while True:
    print("-"*20)
    # turker_response =
    state = env.reset(mode=MODE) # turker_response
    sys_act = None # initial sys act
    total_rewards = 0
    while True:
        print(state)
        print(env.system.state)
        action = pg_reinforce.sampleAction(state[np.newaxis, :], interactive=True)
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
        pg_reinforce.storeRollout(state, action, reward)

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


