import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import random
import pdb

from simulator.nlg_model.nlg_config import Config
config = Config()

def cuda_(var):
    return var.cuda() if config.use_gpu else var

def toss_(p):
    return random.randint(0, 99) <= p

def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)


class NLG_model(nn.Module):
    def __init__(self, data_processor):
        super().__init__()

        self.teacher_force = config.teacher_force
        self.hidden_size = config.hidden_size
        pad_token_id = [data_processor.TARGET.vocab.stoi[token] for token in data_processor.TARGET.vocab.stoi if token == '<pad>'][0]
        self.dec_loss = nn.NLLLoss(ignore_index=pad_token_id)
        self.topk = config.topk


        self.data_processor = data_processor
        self.train_iter, self.valid_iter, self.test_iter = self.data_processor.train_iter, \
                                                           self.data_processor.valid_iter, \
                                                           self.data_processor.test_iter
        self.embed_size = data_processor.TARGET.vocab.vectors.shape[1]
        self.vocab_size = data_processor.TARGET.vocab.vectors.shape[0]
        self.gru_enc = nn.GRU(self.embed_size, config.hidden_size, config.n_layers, dropout=config.dropout, bidirectional=True)
        self.gru_dec = nn.GRU(self.embed_size, config.hidden_size, config.n_layers, dropout=config.dropout, bidirectional=True)
        # self.label = cuda_(nn.Linear(config.hidden_size*2, config.num_actions))
        self.word_embeddings, embedding_dim = self._load_embeddings(self.vocab_size, use_pretrained_embeddings=True,
                                                                    embeddings=data_processor.TARGET.vocab.vectors)

        # attention
        self.attn = nn.Linear(self.hidden_size * 2, config.max_utt_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.proj = nn.Linear(self.hidden_size*2, self.vocab_size)
        self.proj_copy1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.proj_copy2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        # self.dropout_rate = dropout_rate

        init_gru(self.gru_enc)
        init_gru(self.gru_dec)
        self.gru_enc = cuda_(self.gru_enc)
        self.gru_dec = cuda_(self.gru_dec)

    def _load_embeddings(self, vocab_size, emb_dim=None, use_pretrained_embeddings=False, embeddings=None):
        """Load the embeddings based on flag"""

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")

        if not use_pretrained_embeddings:
            word_embeddings = cuda_(torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0))

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)

        word_embeddings = cuda_(word_embeddings)
        return word_embeddings, emb_dim

    def forward(self, input_seqs, input_lens, target_seqs, target_lens=None, hidden=None, is_decode=False):
        # seq2seq implementation

        # z_enc_out, (16, 1, 50) (t, b, h)
        # z_copy_score (b, v)
        ############# encode ##################
        input_seqs = cuda_(input_seqs.transpose(0, 1))
        x_embed = self.word_embeddings(input_seqs)
        x_embed = x_embed.transpose(0, 1)
        sort_idx = np.argsort(-input_lens)
        unsort_idx = cuda_(torch.LongTensor(np.argsort(sort_idx)))
        input_lens = cuda_(input_lens[sort_idx])
        sort_idx = cuda_(torch.LongTensor(sort_idx))
        x_embed = x_embed[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(x_embed, input_lens.tolist())
        packed = cuda_(packed)
        outputs_from_enc, hidden = self.gru_enc(packed, hidden)

        outputs_from_enc, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs_from_enc)
        outputs_from_enc = torch.cat([outputs_from_enc[:, :, :config.hidden_size], outputs_from_enc[:, :, :config.hidden_size]], dim=2)#outputs[:, :, :config.hidden_size] + outputs[:, :, config.hidden_size:]
        outputs_from_enc = outputs_from_enc.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        # self.hidden = hidden
        # print(hidden.shape)
        # hidden = torch.cat([hidden.view(config.n_layers, 2, -1, config.hidden_size)[config.n_layers-1][0],
        #                       hidden.view(config.n_layers, 2, -1, config.hidden_size)[config.n_layers-1][1]], 1)

        # z_enc_out is otputs

        # outputs_from_enc [7, 64, 400])
        ############# decode ##################
        last_hidden = hidden
        import pdb
        pdb.set_trace()
        if not is_decode:
            prev_token = target_seqs[0].view(1, -1)            # prev_token = target_seqs[:, 0].view(1, -1)
        else:
            prev_token = target_seqs
        pm_dec_proba = []
        m_dec_outs = []
        decoded = []
        # if is_decode:
        #     decoded = []

        for t in range(config.max_utt_len):
            teacher_forcing = toss_(self.teacher_force)
            results = self.run_one_token(outputs_from_enc=outputs_from_enc,
                                         prev_token=prev_token,
                                         last_hidden=last_hidden,
                                         is_decode=is_decode,
                                         input_seqs=input_seqs)
            if not is_decode:
                proba, last_hidden, dec_out, log_proba = results
            else:
                predict_token = results.clone()
                decoded.append(predict_token)

            if t != (config.max_utt_len - 1):
                if not is_decode:
                    if teacher_forcing:
                        prev_token = target_seqs[t+1].view(1, -1)# prev_token = target_seqs[:, t+1].view(1, -1)
                    else:
                        # _, prev_token = torch.topk(proba, self.topk)
                        # prev_token = prev_token.view(1, -1)
                        # prev_token = cuda_(Variable(prev_token))

                        mt_proba, mt_index = torch.topk(proba, self.topk)  # [B, topk], [B, topk]
                        selected_id = torch.multinomial(mt_proba, 1).squeeze()
                        prev_token = torch.tensor([mt_index[i][word_id] for i, word_id in enumerate(selected_id)],
                                                  dtype=torch.int64, requires_grad=False)
                        prev_token = cuda_(Variable(prev_token))

                else:
                    prev_token = cuda_(Variable(results).view(1, -1))
                # prev_token = cuda_(Variable(prev_token))

            if not is_decode:
                pm_dec_proba.append(log_proba)
                m_dec_outs.append(dec_out)
        if not is_decode:
            pm_dec_proba = torch.stack(pm_dec_proba, dim=0)

        if is_decode:
            return decoded
        else:
            return pm_dec_proba

    def run_one_token(self, outputs_from_enc, prev_token, input_seqs, last_hidden=None, is_decode=False):
        target_embed = self.word_embeddings(prev_token)  # <go> embedding, torch.Size([64, 50])
        # target_embed = target_embed.unsqueeze(0)
        import pdb
        # pdb.set_trace()

        # attn_weights = F.softmax(
        #     self.attn(torch.cat((target_embed[0], last_hidden[0]), 1)), dim=1)
        # attn_applied = torch.bmm(attn_weights.unsqueeze(0),
        #                          outputs_from_enc.unsqueeze(0))
        #
        # output = torch.cat((target_embed[0], attn_applied[0]), 1)
        # output = self.attn_combine(output).unsqueeze(0)
        #
        # output = F.relu(output)




        gru_out, last_hidden = self.gru_dec(target_embed, last_hidden) # gru_out ()
        gen_score = self.proj(gru_out).squeeze(0) #(b, v)
        # z_copy_score = torch.tanh(self.proj_copy2(outputs_from_enc.transpose(0, 1))) # (b, t, h)
        # z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2) # (b, t)
        # z_copy_score = z_copy_score.cpu() # (b, t)
        # z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0] # (b, t)
        # z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
        # z_copy_score = torch.log(z_copy_score).squeeze(
        #     1) + z_copy_score_max  # [B,V], sparse_z_input [b, t, v+t]
        # z_copy_score = cuda_(z_copy_score)
        #
        # scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1) #[b, 2v]
        # gen_score, z_copy_score = scores[:, :self.vocab_size], \
        #                           scores[:, self.vocab_size:]
        # proba = gen_score + z_copy_score[:, :self.vocab_size]  # [B,V]
        # proba = torch.cat([proba, z_copy_score[:, self.vocab_size:]], 1)

        proba = F.softmax(gen_score, dim=1)
        log_proba = F.log_softmax(gen_score, dim=1)
        if not is_decode:
            return proba, last_hidden, gru_out, log_proba
        else:
            import pdb
            # pdb.set_trace()
            mt_proba, mt_index = torch.topk(proba, self.topk)  # [B, topk], [B, topk]
            selected_id = torch.multinomial(mt_proba, 1).squeeze(1)
            mt_index_1 = torch.tensor([mt_index[i][word_id] for i, word_id in enumerate(selected_id)],
                                      dtype=torch.int64, requires_grad=False)
            # mt_index = mt_index.data.view(-1
            # if input_seqs is None:
            #     pass# tmp = u_input_np  # [,B]
            # else:
            #     # pdb.set_trace()
            #     tmp = input_seqs
            #
            # for i in range(mt_index.size(0)):
            #     # if mt_index[i] >= cfg.vocab_size:
            #     #     mt_index[i] = 2  # unk
            #     if mt_index[i] >= self.vocab_size:
            #         # print(z_index)
            #         mt_index[i] = torch.tensor(int(tmp[mt_index[i] - self.vocab_size, i]))
            #
            # # one_token_decoded = mt_index.clone()
            return mt_index_1

    def supervised_loss(self, pm_dec_proba, m_input):
        pm_dec_proba = pm_dec_proba
        pm_dec_proba = pm_dec_proba[:, :, :self.vocab_size].contiguous()
        # pr_loss = self.pr_loss(pz_proba.view(-1, pz_proba.size(2)), z_input.view(-1))
        m_loss = self.dec_loss(pm_dec_proba.view(-1, pm_dec_proba.size(2)), m_input.view(-1))

        loss = m_loss#pr_loss + m_loss
        return loss#, pr_loss, m_loss

    def test(self, input_seqs, input_seq_lens):
        go_token_id = self.data_processor.TARGET.vocab.stoi['<go>']
        target_seqs = torch.tensor([[go_token_id] * config.batch_size], dtype=torch.int64)
        decoded = self.forward(input_seqs, input_seq_lens, target_seqs,
                      target_lens=None, hidden=None, is_decode=True)

        decoded = torch.stack(decoded, dim=0).transpose(1, 0)
        sents = []
        for one_decoded in decoded:
            one_sent = []
            for i in one_decoded:
                if i.item() == self.data_processor.TARGET.vocab.stoi['<eos>']:
                    break
                token = self.data_processor.TARGET.vocab.itos[i.item()]
                one_sent.append(token)
            sents.append(" ".join(one_sent))

        return sents

    def greedy_decode(self, pz_dec_outs, u_enc_out, m_tm1, u_input_np, last_hidden, degree_input, bspan_index):



        decoded = []
        probas = []
        bspan_index_np = pad_sequences(bspan_index).transpose((1, 0))
        for t in range(self.max_ts):
            proba, last_hidden, _ = self.m_decoder(pz_dec_outs, u_enc_out, u_input_np, m_tm1,
                                                   degree_input, last_hidden, bspan_index_np)
            probas.append(proba)
            mt_proba, mt_index = torch.topk(proba, 1)  # [B,1]
            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())

            if prev_z_input_np is None:
                tmp = u_input_np  # [,B]
            else:
                # pdb.set_trace()
                tmp = np.concatenate((u_input_np, prev_z_input_np), axis=0)


            for i in range(mt_index.size(0)):
                # if mt_index[i] >= cfg.vocab_size:
                #     mt_index[i] = 2  # unk
                if mt_index[i] >= cfg.vocab_size:
                    # print(z_index)
                    mt_index[i] = torch.tensor(int(tmp[mt_index[i] - cfg.vocab_size, i]))


            mt_index = mt_index.data.view(-1)
            decoded.append(mt_index.clone())


            m_tm1 = cuda_(Variable(mt_index).view(1, -1))
        decoded = torch.stack(decoded, dim=0).transpose(0, 1)
        decoded = list(decoded)
        return [list(_) for _ in decoded], probas
