import sys
sys.path.append('/home/wyshi/simulator/')
from simulator.multiwoz_utils import delexicalize
from simulator.nlu_model.main_nlu_train import load_nlu_model, single_pred
from config import Config
config = Config()


class usr_act_predictor(object):
    def __init__(self):

        self.nlu_model = load_nlu_model(config.rule_base_sys_nlu)

    def predict(self, usr_sent):
        delex_sent, kv_dic = delexicalize.delexicalize_one_sent(usr_sent)
        usr_act_str = single_pred(self.nlu_model, delex_sent)[0].lower()

        return usr_act_str