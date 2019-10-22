import torch
import pickle as pkl

class Config():
    f_path = '/home/wyshi/simulator/data/multiwoz-master/data/multi-woz/nlg/'
    vector_cache_path = '/home/wyshi/simulator/data/multiwoz-master/data/multi-woz/vector_cache1'
    # feature_name = ['utt']
    # label_name = 'y'
    use_gpu = False#torch.cuda.is_available()

    # pad token
    pad_token_id = 1

    teacher_force = 100
    topk = 5

    # csv label name
    source_name = 'source'
    target_name = 'target'


    batch_size = 64
    hidden_size = 200
    n_layers = 2
    dropout = 0.3
    max_utt_len = 25

    num_epochs = 30

    # with open('data/multiwoz-master/data/multi-woz/nlu/labelEncoder.pkl', 'rb') as fh:
    #     le = pkl.load(fh)
    # num_actions = len(le.classes_)

    model_save_dir = "/home/wyshi/simulator/simulator/nlg_model/model/model-test-30-new.pkl"