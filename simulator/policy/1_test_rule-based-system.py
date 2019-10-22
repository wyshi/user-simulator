import sys
sys.path.append("/home/wyshi/simulator")
from simulator.user import User, Goal
from simulator.system import System
from simulator.env import Enviroment
import simulator.dialog_config as dialog_config
import numpy as np
from simulator.agent.core import SystemAct
from simulator.policy.config_1 import Config

config = Config()
MODE = dialog_config.INTERACTIVE
if MODE == dialog_config.INTERACTIVE:
    config.INTERACTIVE = True


user = User(nlg_sample=False, nlg_template=True)
system = System(config=config)
env = Enviroment(user=user, system=system, verbose=True, config=config)
sys_act = None
status = []

for _ in range(1):
    print("-"*20)
    # turker_response =
    next_state = env.reset(mode=MODE) # turker_response
    sys_act = None # initial sys act
    total_rewards = 0
    while True:
        provided_sys_act = None
        result_step_sys = env.step_system(provided_sys_act=provided_sys_act, mode=MODE)
        assert result_step_sys is None
        if result_step_sys is not None:
            # goes into FAILED_DIALOG, shouldn't happen in rule_policy and INTERACTIVE mode
            next_state, reward, env.done = result_step_sys
            sys_sent = "Sorry, an error message occurred."#env.last_sys_sent
        else:
            sys_sent = env.last_sys_sent
            print(sys_sent)

            # turker_response =
            next_state, reward, env.done = env.step_user(mode=MODE, mturk_res=None) # turker_response

        total_rewards += reward

        if env.done:
            sys_sent = env.last_sys_sent
            print(sys_sent)
            status.append(env.success)
            # assert env.success
            print('dialog_status: {}'.format(env.success))
            print('reward: {}'.format(total_rewards))
            print("-"*20)
            break


