from RL.simulator.env import Enviroment
from RL.simulator.simulator import User
from RL.simulator.agent import Agent
from numpy import random

import numpy as np
import pickle as pkl

with open("/Users/wyshi/Box Sync/nlp/research/VAE/data/results/transition_prob.pkl", "r") as fh:
    transition_prob = pkl.load(fh)

# area, food, pricerange, phone, address, postcode, name
with open("/Users/wyshi/Box Sync/nlp/research/VAE/data/cambridge_data/state_by_slots_no_dontcare", "r") as fh:
    slots_by_state = pkl.load(fh)

actionID_to_template = {
    0: "thank you for using our system, goodbye .", # arrival
    1: "do you have a [slot_food] preference ?", # food == 0
    2: "do you have a [slot_area] preference ?", # area == 0
    3: "do you have a [slot_pricerange] preference ?", # pricerange == 0
    4: "I am sorry, but there are no restaurants matching your request. Is there anything else I can help you with?",# queryable and no match and presented == 0
    5: "[value_name] is a good restaurant matching your request. Is there anything else I can help you with?", # queryable and match and presented == 0, if "anything else", presented <= match_nums, match_num>0
    6: "[value_name] is located at [value_address] . its [slot_phone] is [value_phone] and the [slot_postcode] is [slot_postcode] . is there anything else i can help you with ?" # address == 1 and presented == 1
}

user = User(transition_prob, slots_by_state, num_action=7, is_sample=True)
env = Enviroment(state_dim=10, num_action=7, num_entity=8, user=user, agent=None)
agent = Agent(env)

state = env.reset()
total_rewards = 0
#prev_rnn_states = [np.zeros((1, HIDDEN_DIM)), np.zeros((1, HIDDEN_DIM))]
bit_vector = [0, 1, 1, 1, 0, 0, 0]
dialog_len = 0
MAX_STEPS = 10
SAMPLE = False
dialogs = []
states = []
cur_dialog = []

start_transition_counts_simulated = np.zeros((7))
transition_counts_simulated = np.zeros((7, 7))
import copy
for dialog_i in xrange(200):
    state = env.reset()
    action = "START"
    #cur_dialog = []
    cur_states = []
    total_rewards = 0
    prev_action_id = -1
    cur_dialog = []
    for t in range(MAX_STEPS):
        cur_turn = []
        dialog_len += 1
        # next_state, reward, done = env.step(action, SAMPLE)
        # env.racender()
        if env.queryable():
            bit_vector[4], bit_vector[5] = env.query_status()
        # print(bit_vector)
        print(state)
        print("User: {}".format(env.user_action))
        print(user.states_series[-1])
        # action = random.choice(range(1, 7), size=1)[0]
        # action = actionID_to_template[action]
        action_id, action = agent.respond(env.user_action, state)
        if prev_action_id != -1:
            transition_counts_simulated[prev_action_id, action_id] += 1
        else:
            start_transition_counts_simulated[action_id] += 1
        print("System: {}".format(action))
        #cur_dialog += ["User: {}\n".format(env.user_action), "System: {}\n".format(action)]
        cur_turn = ("conv_"+str(dialog_i), env.user_action, action)
        cur_dialog.append(cur_turn)
        next_state, reward, done = env.step(action, SAMPLE)
        cur_states += [env.state_maintained]

        # total_rewards += reward
        # reward = -10 if done else reward # normalize reward

        total_rewards += reward
        # pg_reinforce.storeRollout(state, action, reward, prev_rnn_states, bit_vector)

        state = next_state
        prev_action_id = action_id
        print("turn end----")
        if done: break
    print(env.user_action)
    print(total_rewards)
    print(state)
    print("------------------------------")
    # cur_dialog += ["User: "+env.user_action+"\n"]
    # cur_dialog += ["-"*30+"\n\n\n"]
    dialogs.append(cur_dialog)
    states.append(cur_states)

data_simulated = {'train': copy.deepcopy(dialogs)}
data_simulated['test'] = copy.deepcopy(dialogs)

with open("/Users/wyshi/Desktop/nlp/research/VAE/data/simulated/data_simulated.pkl", "w") as fh:
    pkl.dump(data_simulated, fh)

api_simulated = SWDADialogCorpus("/Users/wyshi/Desktop/nlp/research/VAE/data/simulated/data_simulated.pkl", word2vec="/Users/wyshi/Desktop/DeepLearning/CNN_classification/CNN_sentence/GoogleNews-vectors-negative300.bin.gz", word2vec_dim=300)

with open("/Users/wyshi/Desktop/nlp/research/VAE/data/simulated/api_simulated.pkl", "w") as fh:
    pkl.dump(api_simulated, fh)

with open("/Users/wyshi/Desktop/nlp/research/VAE/RL/data/simulated_dialogs_new_with_agent_500", "w") as fh:
    fh.writelines(cur_dialog)


transition_prob_simulated = np.zeros((7, 7))
for i in range(7):
    row_sum = transition_counts_simulated[i, :].sum()
    if row_sum:
        transition_prob_simulated[i] = transition_counts_simulated[i]/row_sum


with open("/Users/wyshi/Desktop/nlp/research/VAE/RL/data/simulated_transition_prob.pkl", "w") as fh:
    pkl.dump(transition_prob_simulated, fh)

with open("/Users/wyshi/Desktop/nlp/research/VAE/RL/data/start_transition_prob_simulated.pkl", "w") as fh:
    pkl.dump(start_transition_prob_simulated, fh)

with open("/Users/wyshi/Desktop/nlp/research/VAE/RL/data/simulated_transition_prob.pkl", "r") as fh:
    transition_prob_simulated_1 = pkl.load(fh)

transition_prob_simulated_1 = np.array(transition_prob_simulated)

start_transition_prob_simulated = start_transition_counts_simulated/sum(start_transition_counts_simulated)


for dialog_i in xrange(10):
    state = env.reset()
    action = "START"
    #cur_dialog = []
    cur_states = []
    total_rewards = 0
    prev_action_id = -1
    cur_dialog = []
    for t in range(MAX_STEPS):
        cur_turn = []
        dialog_len += 1
        # next_state, reward, done = env.step(action, SAMPLE)
        # env.racender()
        if env.queryable():
            bit_vector[4], bit_vector[5] = env.query_status()
        # print(bit_vector)
        print(state)
        print("User: {}".format(env.user_action))
        print(user.states_series[-1])
        # action = random.choice(range(1, 7), size=1)[0]
        # action = actionID_to_template[action]
        action_id, action = agent.respond(env.user_action, state)
        if prev_action_id != -1:
            transition_counts_simulated[prev_action_id, action_id] += 1
        else:
            start_transition_counts_simulated[action_id] += 1
        print("System: {}".format(action))
        #cur_dialog += ["User: {}\n".format(env.user_action), "System: {}\n".format(action)]
        cur_turn = ("conv_"+str(dialog_i), env.user_action, action)
        cur_dialog.append(cur_turn)
        next_state, reward, done = env.step(action, SAMPLE)
        cur_states += [env.state_maintained]

        # total_rewards += reward
        # reward = -10 if done else reward # normalize reward

        total_rewards += reward
        # pg_reinforce.storeRollout(state, action, reward, prev_rnn_states, bit_vector)

        state = next_state
        prev_action_id = action_id
        print("turn end----")
        if done: break
    print(env.user_action)
    print(total_rewards)
    print(state)
    print("------------------------------")
    # cur_dialog += ["User: "+env.user_action+"\n"]
    # cur_dialog += ["-"*30+"\n\n\n"]
    dialogs.append(cur_dialog)
    states.append(cur_states)