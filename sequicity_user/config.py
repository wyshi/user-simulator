import logging
import time
import configparser

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 4
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.6

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'
        self.dataset = 'unknown'

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'tsdf-camrest':self._camrest_tsdf_init,
            'tsdf-kvret':self._kvret_tsdf_init,

            'tsdf-sys': self._sys_tsdf_init,
            'tsdf-usr': self._usr_tsdf_init,
            'tsdf-usr_act' : self._usr_act_tsdf_init
        }
        init_method[m]()

    def _camrest_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 8
        self.max_ts = 40
        self.early_stop_count = 3
        self.cuda = True


        self.vocab_path = './vocab/vocab-camrest.pkl'
        self.data = './data/CamRest676/CamRest676.json'
        self.entity = './data/CamRest676/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRestDB.json'
        self.model_path = './models/camrest.pkl'
        self.result_path = './results/camrest-rl.csv'

        self.glove_path = '../sequicity/data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _sys_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True

        self.split = (9, 1, 1)
        self.model_path = './models/multiwoz_sys911.pkl'
        self.result_path = './results/multiwoz_sys.csv'
        self.vocab_path = './vocab/vocab-multiwoz_sys.pkl'

        self.data = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_sys.json'
        self.entity = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant_db.json'


        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/data/qkun/sequicity/data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _usr_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True
        self.degree_size = 1



        # self.split = (9, 1, 1)
        # self.model_path = './models/multi_woz_simulator911_goal.pkl'
        # self.result_path = './results/multi_woz_simulator911_goal.csv'
        # self.vocab_path = './vocab/vocab-multi_woz_simulator911_goal.pkl'

        # self.data = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_usr_simulator.json'
        # self.entity = './data/multi_woz/rest_OTGY.json'
        # self.db = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant_db.json'


        self.split = (9, 1, 1)
        self.root_dir = "/data/qkun/sequicity_multiwoz_0.4"
        self.model_path = self.root_dir + '/models/multi_woz_simulator911_goal.pkl'
        self.result_path = self.root_dir + '/results/multi_woz_simulator911_goal.csv'
        self.vocab_path = self.root_dir + '/vocab/vocab-multi_woz_simulator911_goal.pkl'

        self.data = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_usr_simulator_goalkey.json'
        self.entity = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant_db.json'

        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/data/qkun/sequicity/data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _usr_act_tsdf_init(self):
        self.vocab_size = 800
        self.embedding_size = 50
        self.hidden_size = 50
        self.lr = 0.003
        self.lr_decay = 0.5
        self.layer_num = 1
        self.z_length = 16
        self.max_ts = 50
        self.early_stop_count = 5
        self.cuda = True
        self.degree_size = 1

        self.split = (9, 1, 1)
        self.root_dir = "/data/qkun/sequicity_multiwoz_0.4"
        self.model_path = self.root_dir + '/models/multi_woz_simulator911_act3.pkl'
        self.result_path = self.root_dir + '/results/multi_woz_simulator911_act.csv'
        self.vocab_path = self.root_dir + '/vocab/vocab-multi_woz_simulator911_act3.pkl'

        self.data = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_usr_simulator_act.json'
        self.entity = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/rest_OTGY.json'
        self.db = '/data/qkun/simulator/data/multiwoz-master/data/multi-woz/restaurant_db.json'

        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.glove_path = '/data/qkun/sequicity/data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.spv_proportion = 100
        self.new_vocab = True
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False


    def _kvret_tsdf_init(self):
        self.prev_z_method = 'separate'
        self.intent = 'all'
        self.vocab_size = 1400
        self.embedding_size = 50
        self.hidden_size = 50
        self.split = None
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-kvret.pkl'
        self.train = './data/kvret/kvret_train_public.json'
        self.dev = './data/kvret/kvret_dev_public.json'
        self.test = './data/kvret/kvret_test_public.json'
        self.entity = './data/kvret/kvret_entities.json'
        self.glove_path = './data/glove/glove.6B.50d.txt'
        self.batch_size = 32
        self.degree_size = 5
        self.z_length = 8
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.rl_epoch_num = 2
        self.cuda = False
        self.spv_proportion = 100
        self.alpha = 0.0
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/kvret.pkl'
        self.result_path = './results/kvret.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

