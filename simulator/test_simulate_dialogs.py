from simulator.env import Enviroment
from simulator.simulator import User
from numpy import random
import pickle as pkl

with open("data/results/transition_prob.pkl", "r") as fh:
    transition_prob = pkl.load(fh)

# area, food, pricerange, phone, address, postcode, name
with open("data/cambridge_data/state_by_slots_no_dontcare", "rb") as fh:
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

for _ in range(1000):
    state = env.reset()
    action = "START"
    #cur_dialog = []
    cur_states = []
    total_rewards = 0
    for t in range(MAX_STEPS):
        dialog_len += 1
        # next_state, reward, done = env.step(action, SAMPLE)
        # env.racender()
        if env.queryable():
            bit_vector[4], bit_vector[5] = env.query_status()
        # print(bit_vector)
        print("User: {}".format(env.user_action))
        print(user.states_series[-1])
        action = random.choice(range(1, 7), size=1)[0]
        action = actionID_to_template[action]
        # action, prev_rnn_states = pg_reinforce.sampleAction(state[np.newaxis ,:], prev_rnn_states, bit_vector)
        # print(prev_rnn_states)
        print("System: {}".format(action))
        cur_dialog += ["User: {}\n".format(env.user_action), "System: {}\n".format(action)]
        next_state, reward, done = env.step(action, SAMPLE)
        cur_states += [env.state_maintained]

        # total_rewards += reward
        # reward = -10 if done else reward # normalize reward

        total_rewards += reward
        # pg_reinforce.storeRollout(state, action, reward, prev_rnn_states, bit_vector)

        state = next_state
        if done: break
    print(env.user_action)
    print(total_rewards)
    print("------------------------------")
    cur_dialog += ["User: "+env.user_action+"\n"]
    cur_dialog += ["-"*30+"\n\n\n"]
    dialogs.append(cur_dialog)
    states.append(cur_states)

