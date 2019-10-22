import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from config import Config
from simulator.nlu_model.data_preprocess import DataProcessor, l2_matrix_norm, AverageMeter, print_cm, eval_
import numpy as np
import pdb
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
import pickle as pkl

# config = Config()

def cuda_(var, config):
    return var.cuda() if config.use_gpu else var

def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)

class Net(nn.Module):
    def __init__(self, state_dim, num_actions, config):
        super(Net, self).__init__()
        # import pdb
        # pdb.set_trace()
        self.config = config
        if config.use_sent:

            self.data_processor = DataProcessor()

            self.embed_size = self.data_processor.TEXT.vocab.vectors.shape[1]
            self.vocab_size = self.data_processor.TEXT.vocab.vectors.shape[0]
            self.gru = nn.GRU(self.embed_size, config.hidden_size, config.n_layers, dropout=config.dropout, bidirectional=True)
            self.word_embeddings, embedding_dim = self._load_embeddings(self.vocab_size, use_pretrained_embeddings=True,
                                                                        embeddings=self.data_processor.TEXT.vocab.vectors)

            init_gru(self.gru)
            self.gru = cuda_(self.gru, config=self.config)

            self.linear1 = nn.Linear(state_dim + config.hidden_size * 2, config.hidden_size)
            self.linear2 = nn.Linear(config.hidden_size, num_actions)

        elif config.use_sent_one_hot:
            if os.path.exists("data/multiwoz-master/data/multi-woz/nlu/CountVectorizer.pkl"):
                with open("data/multiwoz-master/data/multi-woz/nlu/CountVectorizer.pkl", "rb") as fh:
                    # pkl.dump(fh, cnt_vectorizer)
                    self.cnt_vectorizer = pkl.load(fh)
            else:
                df1 = pd.read_csv("data/multiwoz-master/data/multi-woz/nlu/train_no.csv")
                df2 = pd.read_csv("data/multiwoz-master/data/multi-woz/nlu/valid_no.csv")
                df3 = pd.read_csv("data/multiwoz-master/data/multi-woz/nlu/test_no.csv")
                df = df1.append(df2).append(df3)
                self.cnt_vectorizer = CountVectorizer()
                # pdb.set_trace()
                self.cnt_vectorizer.fit(df['utt'].tolist())
            # pdb.set_trace()
            self.linear1 = nn.Linear(state_dim + len(self.cnt_vectorizer.vocabulary_), 50)
            self.linear2 = nn.Linear(50, num_actions)
        else:
            self.linear1 = nn.Linear(state_dim, 20)
            self.linear2 = nn.Linear(20, num_actions)

        # self.W1 = nn.Parameter(torch.randn(state_dim, 20))
        # self.b1 = nn.Parameter(torch.randn(20))
        # self.W2 = nn.Parameter(torch.randn(20, num_actions))
        # self.b2 = nn.Parameter(torch.randn(num_actions))

        # self.myparameters = nn.ParameterList([nn.Parameter(self.W1), nn.Parameter(self.W2),
        #                                       nn.Parameter(self.b1), nn.Parameter(self.b2)])


    def _load_embeddings(self, vocab_size, emb_dim=None, use_pretrained_embeddings=False, embeddings=None):
        """Load the embeddings based on flag"""

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")

        if not use_pretrained_embeddings:
            word_embeddings = cuda_(torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0), config=self.config)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)

        word_embeddings = cuda_(word_embeddings, config=self.config)
        return word_embeddings, emb_dim

    def forward(self, states, bit_vecs=None):
        # pdb.set_trace()
        if self.config.use_sent:
            return self.forward_sent(states=states, bit_vecs=bit_vecs, hidden=None)
        elif self.config.use_sent_one_hot:
            return self.forward_sent_one_hot(states=states, bit_vecs=bit_vecs)
        else:
            return self.forward_no_sent(states=states, bit_vecs=bit_vecs)

    def forward_sent(self, states, bit_vecs=None, hidden=None):
        np_state, sent = states
        np_state = np_state[np.newaxis, :]
        input_seqs = self.data_processor.process_one_data(sent)
        input_lens = torch.LongTensor([input_seqs.shape[1]])

        input_seqs = cuda_(input_seqs.transpose(0, 1), config=self.config)
        x_embed = self.word_embeddings(input_seqs)
        x_embed = x_embed.transpose(0, 1)
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)), config=self.config)
        input_lens = cuda_(input_lens[sort_idx], config=self.config)
        sort_idx = cuda_(torch.LongTensor(sort_idx), config=self.config)
        x_embed = x_embed[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_embed, input_lens.tolist())
        packed = cuda_(packed, config=self.config)
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = torch.cat([outputs[:, :, :self.config.hidden_size], outputs[:, :, :self.hidden_size]], dim=2)#outputs[:, :, :config.hidden_size] + outputs[:, :, config.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        # self.hidden = hidden
        # print(hidden.shape)
        hidden = torch.cat([hidden.view(self.config.n_layers, 2, -1, self.config.hidden_size)[self.config.n_layers-1][0],
                              hidden.view(self.config.n_layers, 2, -1, self.config.hidden_size)[self.config.n_layers-1][1]], 1)

        # pdb.set_trace()
        state_from_np = torch.from_numpy(np_state).float()#.unsqueeze(0)
        h1 = torch.tanh(self.linear1(torch.cat([state_from_np, hidden], dim=1)))
        p = self.linear2(h1)
        # import pdb
        # pdb.set_trace()
        # p = F.log_softmax(p, dim=1)
        # if bit_vecs :
        #     if not isinstance(bit_vecs, torch.Tensor):
        #         bit_vecs = torch.tensor(bit_vecs, dtype=torch.float32, device=Config.device)
        #         bit_vecs.detach_()
        #     p = p * bit_vecs

        # h1 = F.tanh((torch.matmul(states, self.W1) + self.b1))
        # p = torch.matmul(h1, self.W2) + self.b2
        return p

    def forward_sent_one_hot(self, states, bit_vecs=None):
        if len(states) == 2:
            states, sent = states
            sent_one_hot = self.cnt_vectorizer.transform([sent]).toarray()[0]
            # pdb.set_trace()
            states = np.hstack([states, sent_one_hot])

        states = states[np.newaxis, :]
        states = torch.tensor(states, dtype=torch.float32, device=self.config.device)

        h1 = torch.tanh(self.linear1(states))
        p = self.linear2(h1)
        # p = F.log_softmax(p, dim=1)

        return p

    def forward_no_sent(self, states, bit_vecs=None):
        if len(states) == 2:
            states, sent = states
        states = states[np.newaxis, :]
        states = torch.tensor(states, dtype=torch.float32)

        h1 = torch.tanh(self.linear1(states))
        p = self.linear2(h1)
        # p = F.log_softmax(p, dim=1)



        import pdb
        # pdb.set_trace()
        # if bit_vecs :
        #     if not isinstance(bit_vecs, torch.Tensor):
        #         bit_vecs = torch.tensor(bit_vecs, dtype=torch.float32, device=Config.device)
        #         bit_vecs.detach_()
        #     p = p * bit_vecs

        # h1 = F.tanh((torch.matmul(states, self.W1) + self.b1))
        # p = torch.matmul(h1, self.W2) + self.b2
        return p
