import torch
import os, sys

class Config(object):


    # BEST_MODEL_DIR = "/home/wyshi/simulator/model/save/nlg_sample/oneHot_newReward_bitMore/best/0_2019-5-19-15-13-5-6-139-1.pkl"

    rule_base_sys_nlu = "/home/wyshi/simulator/simulator/nlu_model/model/model-test-30-new.pkl"

    use_sl_simulator = False

    INTERACTIVE = True

    device = 'cpu'#torch.device("cuda:6" if torch.cuda.is_available() else "cpu")#'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = False#torch.cuda.is_available()

    # user simulator
    nlg_sample = False
    nlg_template = False

    # rl
    n_episodes = 30000
    save_dir = '/home/wyshi/simulator/model/save/template/oneHot_newReward_bitMore/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    use_sequicity_for_rl_model = False
    with_bit = True
    with_bit_rep_only = False
    with_bit_more = True
    with_bit_all = False
    if with_bit:
        assert sum([with_bit_rep_only, with_bit_more, with_bit_all]) == 1
    else:
        assert sum([with_bit_rep_only, with_bit_more, with_bit_all]) == 0

    use_new_reward = True


    bit_not_used_in_update = True
    use_sent = False
    use_multinomial = False
    use_sent_one_hot = True
    lr = 1e-4

    discrete_act = True
    discounted_factor = 0.9#0.99#0.9
    init_exp = 0.5 if discrete_act else 0
    final_exp = 0 if discrete_act else 0
    loose_agents = True
    small_value = 0
    warm_start_episodes = 0
    replay = True
    batch_size = 64
    seed = 0
    update_every = 64

    # policy model par
    hidden_size = 200
    n_layers = 2
    dropout = 0.3
    max_utt_len = 25
    num_epochs = 30

    # sequicity parameters
    vocab_size = 800
    pretrained_dir = '/data/qkun/sequicity_multiwoz_0.4/models/multiwoz_sys911.pkl'

