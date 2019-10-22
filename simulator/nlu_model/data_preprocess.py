import pandas as pd
import numpy as np
import numpy as np
import re
import logging
import re
import sys

print(sys.path)
import spacy
from sklearn.metrics import f1_score, precision_score, classification_report, hamming_loss, recall_score

spacy_en = spacy.load('en')
import torch
import torch.nn as nn
from torch.nn import init
from torchtext import vocab
from torchtext import data
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
import pickle as pkl

from simulator.nlu_model.nlu_config import Config
config = Config()

cuda_available = torch.cuda.is_available()


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if cm[i, j] != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable} ||AAT - I||

    Returns:
        regularized value

    """
    sum_res = torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5)
    if cuda_available:
        sum_res = sum_res.float().cuda()
    else:
        sum_res = sum_res.float()
    return sum_res


def eval_(logits, labels, binary=False):
    """
    calculate the accuracy
    :param logits: Variable [batch_size, ]
    :param labels: Variable [batch_size]
    :return:
    """
    if binary is False:
        _, predicted = torch.max(logits.data, 1)
        return (predicted == labels).sum().item()/ labels.size(0)
    else:
        if cuda_available:
            l = torch.ones(logits.size()).cuda()
        else:
            l = torch.ones(logits.size())

        l[logits <= 0] = 0

        return (l == labels).sum().item() / labels.size(0)


def tokenizer(text):  # create a tokenizer function
    # print(text)
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ' and tok.text != '[' and tok.text != ']']
    # print(text)
    tokenized_text = []
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s", "'m", "'re"]
    punctuation = '.,/\\-*&#\'\"'
    num = set('1234567890')
    stop_words = []#['a', 'the', 'of', 'in', 'on', 'to']
    for token in text:
        if token == "n't":
            tmp = 'not'
        if token == 'US':
            tmp = 'America'
        elif token == "'ll":
            tmp = 'will'
        elif token == "'m":
            tmp = 'am'
        elif token == "'s":
            tmp = 'is'
        elif len(set(token) & num) > 0:
            continue
        elif token == "'re":
            tmp = 'are'

        elif token in punctuation:
            continue
        elif token in stop_words:
            continue
        else:
            tmp = token
        tmp = tmp.lower()
        tokenized_text.append(tmp)

    # print(tokenized_text)
    # while len(tokenized_text) < 5:
    #     tokenized_text.append('<pad>')
    return tokenized_text


class BatchWrapper:
    def __init__(self, b_iter, x_var, y_var):
        self.b_iter, self.x_var, self.y_var = b_iter, x_var, y_var

    def __iter__(self):
        for batch in self.b_iter:

            x = getattr(batch, self.x_var)  # we assume only one input in this wrapper
            # t = getattr(batch, self.t_var)
            # h = getattr(batch, self.h_var)
            # neg = getattr(batch, self.neg_var)
            # neu = getattr(batch, self.neu_var)
            # pos = getattr(batch, self.pos_var)
            # stem = getattr(batch, self.stem_var)
            # char = getattr(batch, self.char_var)
            # index = getattr(batch, self.index)
            # personas = []
            # for name in self.personal_names:
            #     personas.append(getattr(batch, name))

            if self.y_var is not None:
                y = getattr(batch, self.y_var)
            else:
                y = torch.zeros((1))
                bool_y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.b_iter)


class custom_Field(data.Field):
    def __init__(self, scope, **kwargs):
        import sys
        from rl.char_embed.encoder import Model
        self.model = Model(scope=scope)
        super(custom_Field, self).__init__(**kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        text_feat = self.model.transform(arr)
        if cuda_available:
            text_feat = torch.from_numpy(text_feat).float().cuda()
        else:
            text_feat = torch.from_numpy(text_feat).float()
        return text_feat


class DataProcessor(object):
    def __init__(self):
        self.f_path = config.f_path
        self.label_name = config.label_name
        self.process_data()
        self.batchlize_data()

    def process_data(self):
        datafields = []
        df = pd.read_csv(self.f_path + 'train_no.csv')
        self.df = df
        self._build_labelEncoder()
        print(df.head(2))
        print(df.columns)

        # Define field
        TEXT = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, lower=True, # eos_token='<EOS>',
                          batch_first=True, truncate_first=True, include_lengths=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        # TURN = data.Field(sequential=False, use_vocab=False)
        # #     IS_LABEL = data.Field(sequential=False, use_vocab=False)
        # INDEX = data.Field(sequential=False, use_vocab=False)
        # HIS = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, lower=True, eos_token='<EOS>',
        #                  batch_first=True, truncate_first=True, include_lengths=True)
        # HIS_STEM = data.Field(sequential=False, truncate_first=True)

        # NEG = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
        #                  postprocessing=data.Pipeline(lambda x, y: float(x)))
        # NEU = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
        #                  postprocessing=data.Pipeline(lambda x, y: float(x)))
        # POS = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
        #                  postprocessing=data.Pipeline(lambda x, y: float(x)))
        # PERSONALS = [data.Field(sequential=False, dtype=torch.float, use_vocab=False,
        #                         postprocessing=data.Pipeline(lambda x, y: float(x)))] * user_embed_dim
        # personal_names = []

        for col in df.columns.tolist():
            if col == 'utt':
                datafields.append((col, TEXT))

            elif col == config.label_name:
                datafields.append((col, LABEL))
            #         elif col == "Index":
            #             datafileds.append((col,IS_LABEL))
            # elif col == 'history':
            #     print("appending history\n\n\n")
            #     datafileds.append((col, HIS))
#
            # elif col == 'Turn':
            #     datafileds.append((col, TURN))
            # elif col == "neg":
            #     datafileds.append((col, NEG))
            # elif col == "neu":
            #     datafileds.append((col, NEU))
            # elif col == "pos":
            #     datafileds.append((col, POS))
            # elif col == "his_stem":
            #     datafileds.append((col, HIS_STEM))
            # elif col[-2:] == '.x':
            #     #         elif col in persona_list:
            #     datafileds.append((col, PERSONALS[cnt]))
            #     personal_names.append(col)
            #     cnt += 1
#
            # else:
            #     datafileds.append((col, None))

        # Define iterator
        self.datafields = datafields
        train, valid = data.TabularDataset.splits(
            format='csv',
            skip_header=True,
            path=self.f_path,
            train='train_no.csv',
            validation='valid_no.csv',
            fields=datafields
        )
        test = data.TabularDataset(
            path=self.f_path + 'test_no.csv',
            format='csv',
            skip_header=True,
            fields=datafields
        )

        # using the training corpus to create the vocabulary
        vectors = Vectors(name='glove.6B.50d.txt', cache=config.vector_cache_path, unk_init=nn.init.uniform_)
        # vectors.unk_init = nn.init.uniform_
        TEXT.build_vocab(train, valid, test, vectors=vectors, max_size=30000)
        self.train, self.valid, self.test = train, valid, test
        self.TEXT = TEXT
        #     HIS.build_vocab(train,valid,test)
        #     HIS.vocab.load_vectors(vectors='fasttext.en.300d')
        #     TEXT.build_vocab(train,valid,test)#, vectors=vectors, max_size=300000)

        print('num of tokens', len(TEXT.vocab.itos))

        print('most common', TEXT.vocab.freqs.most_common(5))

        print('len(train)', len(train))
        print('len(test)', len(test))

        self.train_iter = data.Iterator(dataset=train, batch_size=config.batch_size, train=True, sort_key=lambda x: len(x.utt),
                                   sort_within_batch=False, repeat=False,
                                   device='cuda:0' if config.use_gpu else -1)

        self.valid_iter = data.Iterator(dataset=valid, batch_size=config.batch_size, train=False, sort_key=lambda x: len(x.utt),
                                 sort_within_batch=False, repeat=False,
                                 device='cuda:0' if config.use_gpu else -1)

        self.test_iter = data.Iterator(dataset=test, batch_size=config.batch_size, train=False, sort_key=lambda x: len(x.utt),
                                  sort_within_batch=False, repeat=False,
                                  device='cuda:0' if config.use_gpu else -1)

        self.num_tokens = len(TEXT.vocab.itos)

    def _build_labelEncoder(self):
        from sklearn.preprocessing import LabelEncoder
        with open('/home/wyshi/simulator/data/multiwoz-master/data/multi-woz/nlu/labelEncoder.pkl', 'rb') as fh:
            self.le = pkl.load(fh)
        self.df['y'] = self.le.fit_transform(self.df[config.label_name])

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        df_tf_idf = self.df
        corpus = df_tf_idf.his_stem.tolist()
        vectorizer = TfidfVectorizer(max_features=config.tfidf_dim)
        vectorizer = vectorizer.fit(corpus)

        HIS_STEM = data.Field(sequential=False, truncate_first=True)
        def numer(arr, device):
            arr = vectorizer.transform(arr).toarray()
            var = torch.tensor(arr, dtype=torch.float, device=device)
            return var

        HIS_STEM.numericalize = numer

    def batchlize_data(self):
        # Add wrapper for interator
        self.train_iter = BatchWrapper(self.train_iter, "utt", self.label_name)
        self.valid_iter = BatchWrapper(self.valid_iter, "utt", self.label_name)
        self.test_iter = BatchWrapper(self.test_iter, "utt", self.label_name)

    def process_one_data(self, text):
        # Input data
        text = self.TEXT.preprocess(text)
        text = [[self.TEXT.vocab.stoi[x] for x in text]]
        x = torch.LongTensor(text)

        if config.use_gpu:
            x = x.cuda()

        return x.view(1, -1)


def load_data_model(model, config, num_classes=6,
                    f_path='data/multiwoz-master/data/multi-woz/nlu/',
                    pred=False, cache_path='data/multiwoz-master/data/multi-woz/vector_cache',
                    label_name='act', scope=""):  #
    # Define hyper params
    print("pred or not {}".format(pred))
    print("label name: {}".format(label_name))

    with_persona = config.with_persona
    use_gpu = config.use_gpu
    label_name = label_name
    tfidf_dim = config.tfidf_dim
    user_embed_dim = config.user_embed_dim  # 5#
    hidden_dim = config.hidden_dim
    n_layers = config.n_layers
    drop = config.drop
    datafileds = []
    df = pd.read_csv(f_path + 'train_no.csv')
    print(df.head(2))
    print(df.columns)
    # Fit tf-idf vector
    from sklearn.feature_extraction.text import TfidfVectorizer
    df_tf_idf = df
    corpus = df_tf_idf.his_stem.tolist()
    vectorizer = TfidfVectorizer(max_features=tfidf_dim)
    vectorizer = vectorizer.fit(corpus)

    # Define field
    TEXT = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, lower=True, eos_token='<EOS>',
                      fix_length=config.max_utt_len,
                      batch_first=True, truncate_first=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    TURN = data.Field(sequential=False, use_vocab=False)
    #     IS_LABEL = data.Field(sequential=False, use_vocab=False)
    INDEX = data.Field(sequential=False, use_vocab=False)
    HIS = data.Field(sequential=True, tokenize=tokenizer, use_vocab=True, lower=True, eos_token='<EOS>',
                     batch_first=True, truncate_first=True, include_lengths=True)
    HIS_STEM = data.Field(sequential=False, truncate_first=True)

    def numer(arr, device):
        arr = vectorizer.transform(arr).toarray()
        var = torch.tensor(arr, dtype=torch.float, device=device)
        return var

    HIS_STEM.numericalize = numer
    print("scope {}".format(scope))
    CHAR_FEAT = custom_Field(scope=scope, sequential=False, use_vocab=False)
    NEG = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
                     postprocessing=data.Pipeline(lambda x, y: float(x)))
    NEU = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
                     postprocessing=data.Pipeline(lambda x, y: float(x)))
    POS = data.Field(sequential=False, dtype=torch.float, use_vocab=False,
                     postprocessing=data.Pipeline(lambda x, y: float(x)))
    PERSONALS = [data.Field(sequential=False, dtype=torch.float, use_vocab=False,
                            postprocessing=data.Pipeline(lambda x, y: float(x)))] * user_embed_dim
    personal_names = []
    cnt = 0
    for col in df.columns.tolist():
        if col == 'Unit':
            datafileds.append((col, TEXT))
        elif col == 'Unit_char':
            datafileds.append((col, CHAR_FEAT))
        elif col == 'Index':
            datafileds.append((col, INDEX))

        elif col == label_name:
            datafileds.append((col, LABEL))
        #         elif col == "Index":
        #             datafileds.append((col,IS_LABEL))
        elif col == 'history':
            print("appending history\n\n\n")
            datafileds.append((col, HIS))

        elif col == 'Turn':
            datafileds.append((col, TURN))
        elif col == "neg":
            datafileds.append((col, NEG))
        elif col == "neu":
            datafileds.append((col, NEU))
        elif col == "pos":
            datafileds.append((col, POS))
        elif col == "his_stem":
            datafileds.append((col, HIS_STEM))
        elif col[-2:] == '.x':
            #         elif col in persona_list:
            datafileds.append((col, PERSONALS[cnt]))
            personal_names.append(col)
            cnt += 1

        else:
            datafileds.append((col, None))
    print('personal info dim', cnt)

    # Define iterator
    train, valid = data.TabularDataset.splits(
        format='csv',
        skip_header=True,
        path=f_path,
        train='train_no.csv',
        validation='valid_no.csv',
        fields=datafileds,
    )
    test = data.TabularDataset(
        path=f_path + 'test_no.csv',
        format='csv',
        skip_header=True,
        fields=datafileds,
    )

    # using the training corpus to create the vocabulary
    vectors = Vectors(name='wiki.en.vec', cache=cache_path, unk_init=nn.init.uniform_)
    # vectors.unk_init = nn.init.uniform
    HIS.build_vocab(train, valid, test, vectors=vectors, max_size=30000)
    #     HIS.build_vocab(train,valid,test)
    #     HIS.vocab.load_vectors(vectors='fasttext.en.300d')
    #     TEXT.build_vocab(train,valid,test)#, vectors=vectors, max_size=300000)
    TEXT.vocab = HIS.vocab

    print('num of tokens', len(TEXT.vocab.itos))
    print('num of tokens', len(HIS.vocab.itos))

    print(TEXT.vocab.freqs.most_common(5))
    print(HIS.vocab.freqs.most_common(5))

    print('len(train)', len(train))
    print('len(test)', len(test))

    train_iter = data.Iterator(dataset=train, batch_size=64, train=True, sort_key=lambda x: len(x.Unit),
                               sort_within_batch=False, repeat=False, device=torch.device('cuda:0') if use_gpu else -1)

    val_iter = data.Iterator(dataset=valid, batch_size=256, train=False, sort_key=lambda x: len(x.Unit),
                             sort_within_batch=False, repeat=False, device=torch.device('cuda:0') if use_gpu else -1)

    test_iter = data.Iterator(dataset=test, batch_size=256, train=False, sort_key=lambda x: len(x.Unit),
                              sort_within_batch=False, repeat=False, device=torch.device('cuda:0') if use_gpu else -1)

    num_tokens = len(HIS.vocab.itos)

    print("No .class", num_classes)
    print("with persona: {}".format(with_persona))
    # Define model
    nets = model(vocab_size=num_tokens, embedding=TEXT.vocab.vectors, hidden_dim=hidden_dim, output_dim=num_classes,
                 n_layers=n_layers, bidirectional=True, dropout=drop, tfidf_dim=tfidf_dim,
                 user_embed_dim=user_embed_dim,
                 add_persona=with_persona)

    # Add wrapper for interator
    train_iter = BatchWrapper(train_iter, "Unit", 'Turn', "history", label_name, "neg", "neu", "pos", 'his_stem',
                              'Unit_char', 'Index', personal_names)
    valid_iter = BatchWrapper(val_iter, "Unit", 'Turn', "history", label_name, "neg", "neu", "pos", 'his_stem',
                              'Unit_char', 'Index', personal_names)
    test_iter = BatchWrapper(test_iter, "Unit", 'Turn', "history", label_name, "neg", "neu", "pos", 'his_stem',
                             'Unit_char', 'Index', personal_names)

    if use_gpu:
        cuda1 = torch.device('cuda:0')
        nets.cuda(device=cuda1)
        if pred is True:
            return TEXT, nets, CHAR_FEAT
        return train_iter, valid_iter, test_iter, TEXT, nets
    else:
        if pred is True:
            return TEXT, nets, CHAR_FEAT
        return train_iter, valid_iter, test_iter, TEXT, nets


if __name__ == "__main__":
    data_processor = DataProcessor()
