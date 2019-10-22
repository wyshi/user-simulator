import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import random
from simulator.nlu_model.nlu_config import Config
config = Config()

def cuda_(var):
    return var.cuda() if config.use_gpu else var

def init_gru(gru):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + gru.hidden_size], gain=1)


class NLU_model(nn.Module):
    def __init__(self, data_processor):
        super().__init__()
        self.dp = data_processor
        self.data_processor = data_processor
        self.train_iter, self.valid_iter, self.test_iter = self.dp.train_iter, self.dp.valid_iter, self.dp.test_iter
        self.embed_size = data_processor.TEXT.vocab.vectors.shape[1]
        self.vocab_size = data_processor.TEXT.vocab.vectors.shape[0]
        self.gru = nn.GRU(self.embed_size, config.hidden_size, config.n_layers, dropout=config.dropout, bidirectional=True)
        self.label = cuda_(nn.Linear(config.hidden_size*2, config.num_actions))
        self.word_embeddings, embedding_dim = self._load_embeddings(self.vocab_size, use_pretrained_embeddings=True,
                                                                    embeddings=data_processor.TEXT.vocab.vectors)

        init_gru(self.gru)
        self.gru = cuda_(self.gru)

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

    def forward(self, input_seqs, input_lens, hidden=None):

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
        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = torch.cat([outputs[:, :, :config.hidden_size], outputs[:, :, :config.hidden_size]], dim=2)#outputs[:, :, :config.hidden_size] + outputs[:, :, config.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        self.hidden = hidden
        # print(hidden.shape)
        hidden = torch.cat([hidden.view(config.n_layers, 2, -1, config.hidden_size)[config.n_layers-1][0],
                              hidden.view(config.n_layers, 2, -1, config.hidden_size)[config.n_layers-1][1]], 1)

        logits = self.label(hidden)

        return logits



class CNN(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, n_layers, bidirectional, dropout, batch_size=64,
                 use_turn=True, padding=(1, 0), stride=2, kernel_heights=[3, 4, 5], in_channels=1, out_channels=150,
                 add_sentiment=True, tfidf_dim=None, user_embed_dim=None):
        super().__init__()

        """
        Arguments
        ---------
        batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_dim)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        embedding_dim : Embedding dimension of GloVe word embeddings
        embedding : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        --------

        """

        self.batch_size = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        self.hidden_dim = len(kernel_heights) * out_channels
        self.use_turn = use_turn

        turn_emb_dim = 10
        self.turn_embeddings = torch.nn.Embedding(50, turn_emb_dim)

        self.word_embeddings, embedding_dim = self._load_embeddings(vocab_size, use_pretrained_embeddings=True,
                                                                    embeddings=embedding)

        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim), stride, padding)
        self.his_conv1 = nn.Conv2d(in_channels, out_channels // 2, (kernel_heights[0], embedding_dim), stride, padding)
        self.his_conv2 = nn.Conv2d(in_channels, out_channels // 2, (kernel_heights[1], embedding_dim), stride, padding)
        self.his_conv3 = nn.Conv2d(in_channels, out_channels // 2, (kernel_heights[2], embedding_dim), stride, padding)
        #         self.his_conv1 = self.conv1
        #         self.his_conv2 = self.conv2
        #         self.his_conv3 = self.conv3

        self.dropout = nn.Dropout(dropout)
        his_dim = len(kernel_heights) * out_channels // 2
        his_stem_dim = 100
        char_dim = 4096

        self.his_fc = nn.Linear(tfidf_dim, his_stem_dim)
        his_stem_dim = his_dim
        self.label = nn.Linear(len(kernel_heights) * (out_channels) + his_stem_dim, 50)
        if user_embed_dim is not None:
            self.user_embed = nn.Linear(user_embed_dim, 10)
        else:
            user_embed_dim = 0

        if add_sentiment is True:
            self.label_char = nn.Linear(char_dim, 50)

            self.label1 = nn.Linear(50 + 3 + turn_emb_dim + 50 + 10, output_dim)
            self.bool_fc = nn.Linear(50 + 3 + turn_emb_dim + 50 + 10, 1)
        else:
            self.label1 = nn.Linear(50, output_dim)
            self.bool_fc = nn.Linear(50, 1)

    def _load_embeddings(self, vocab_size, emb_dim=None, use_pretrained_embeddings=False, embeddings=None):
        """Load the embeddings based on flag"""

        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")

        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")

        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)

        return word_embeddings, emb_dim

    def conv_block(self, input, conv_layer):

        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        # print(conv_out.size())
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        # print(activation.size())
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(
            2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, x, turn=None, his=None, sentiment=None, x_len=None, hs_len=None, his_stem=None, char_emb=None,
                personas=None):

        """
        The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
        whose shape for each batch is (num_seq, embedding_dim) with kernel of varying height but constant width which is same as the embedding_dim.
        We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
        and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
        to the output layers consisting two units which basically gives us the logits for both positive and negative classes.

        Parameters
        ----------
        input_sentences: input_sentences of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.
        logits.size() = (batch_size, output_dim)

        """
        if turn is not None:
            t_embed = self.turn_embeddings(turn)
        if his is not None:
            mode = 'normal'
            # TF-IDF
            if mode == 'TFIDF':
                his_embed = his
                his_embed = self.his_fc(his_embed)
            elif mode == 'MEAN':
                # Mean embed
                his_embed = self.word_embeddings(his)
                his_embed = torch.mean(his_embed, dim=1)
            else:
                his_embed = self.word_embeddings(his)
                his_embed = his_embed.unsqueeze(1)
                max_out1 = self.conv_block(his_embed, self.his_conv1)
                max_out2 = self.conv_block(his_embed, self.his_conv2)
                max_out3 = self.conv_block(his_embed, self.his_conv3)
                his_out = torch.cat((max_out1, max_out2, max_out3), 1)

        if his_stem is not None:
            stem = self.his_fc(his_stem)

        x_embed = self.word_embeddings(x)
        # x_embed.size() = (batch_size, num_seq, embedding_dim)
        x_embed = x_embed.unsqueeze(1)
        # x_embed.size() = (batch_size, 1, num_seq, embedding_dim)
        max_out1 = self.conv_block(x_embed, self.conv1)
        max_out2 = self.conv_block(x_embed, self.conv2)
        max_out3 = self.conv_block(x_embed, self.conv3)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)

        all_out = torch.cat((all_out, his_out), 1)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)

        logits = self.label(fc_in)
        if personas is not None:
            personas = self.user_embed(personas)
        if sentiment is not None:
            char_emb = self.label_char(char_emb)
            #             logits=torch.cat((logits,sentiment,char_emb,t_embed),1)
            logits = torch.cat((logits, sentiment, char_emb, t_embed, personas), 1)

            output = self.label1(logits)
            bool_out = self.bool_fc(logits)
        else:
            output = self.label1(logits)
            bool_out = self.bool_fc(logits)

        return output, bool_out, bool_out
