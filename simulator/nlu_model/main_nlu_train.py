import sys
print(sys.path)
sys.path.append("/home/wyshi/simulator")
from simulator.nlu_model.model import NLU_model
from simulator.nlu_model.data_preprocess import DataProcessor, l2_matrix_norm, AverageMeter, print_cm, eval_

from torch import nn
from torch.autograd import Variable
from torch import optim
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm
import sys
from simulator.nlu_model.nlu_config import Config

from tqdm import tqdm

import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, classification_report, hamming_loss, recall_score
import os

plt.switch_backend('agg')

import json

config = Config()


def load_nlu_model(model_dir="simulator/nlu_model/model/model-test.pkl"):
    model = torch.load(model_dir, map_location='cpu')
    data_processor = DataProcessor()
    net = NLU_model(data_processor=data_processor)
    net.load_state_dict(model)
    net.eval()
    return net

def single_pred(net, sent):

    input_seqs = net.data_processor.process_one_data(sent)
    # input_seqs = Variable(input_seqs)
    input_seq_lens = torch.LongTensor([input_seqs.shape[1]])
    logits = net(input_seqs, input_seq_lens)
    predicted = torch.max(logits, 1)[1].item()
    predicted = config.le.inverse_transform([predicted])
    return predicted


if __name__ == '__main__':
    use_attn = False
    use_turn = True

    config = Config()


    n_label = config.num_actions

    epochs = config.num_epochs
    log_dir = './log/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    label_name = [config.le.classes_[i][:10] for i in range(
        n_label)]  # ["A1,greet","A11,neu","A12,ask","A16,self","A17,pos",'A18,neg', 'A19,off', 'A3,askinfo', 'A5,agree', 'A6,disagree', 'end']#["E2","L3","L1","P1","TF","other"]

    data_processor = DataProcessor()
    net = NLU_model(data_processor=data_processor)
    net.optimizer = optim.Adam(params=net.parameters(), lr=5e-4, weight_decay=1e-3)
    net.lr_scheduler = optim.lr_scheduler.StepLR(net.optimizer, step_size=500, gamma=0.95)
    net.loss_func = nn.CrossEntropyLoss()
    #     bool_loss=nn.BCEWithLogitsLoss()

    # penal = l2_matrix_norm(att@attT - identity)

    best_recall = 0
    best_acc = 0

    train_accs = []
    test_accs = []

    epoch_loss = {'train': [], 'val': []}
    epoch_acc = {'train': [], 'val': []}
    epoch_bool_acc = {'train': [], 'val': []}
    epoch_precision = {'train': [], 'val': []}
    epoch_recall = {'train': [], 'val': []}

    for epoch in range(epochs):

        for phase in ('train', 'val'):
            accs = AverageMeter()
            losses = AverageMeter()
            recalls = AverageMeter()
            precisions = AverageMeter()
            bool_accs = AverageMeter()
            if phase == 'train':
                net.train(True)
                phrase_iter = net.train_iter
                net.lr_scheduler.step()

            else:
                net.eval()
                print("running valid.....")
                phrase_iter = net.valid_iter
            end = time.time()
            for l in tqdm(phrase_iter):

                (input_seqs, input_seq_lens), ys = l

                net.optimizer.zero_grad()  # clear the gradient

                logits = net(input_seqs, input_seq_lens)
                loss = net.loss_func(logits, ys)
                acc = eval_(logits, labels=ys.data.long())
                if phase == 'train':
                    loss.backward()
                    clip_grad_norm(net.parameters(), 10)
                    net.optimizer.step()
                    train_accs.append(acc)

                nsample = input_seqs.size(0)
                accs.update(acc, nsample)
                #                 recalls.update(recall, nsample)

                #                 precisions.update(precision, nsample)
                #                 bool_accs.update(bool_acc, nsample)
                losses.update(loss.item(), nsample)

            elapsed_time = time.time() - end

            print('[{}]\tEpoch: {}/{}\tAcc: {:.2%}\tLoss: {:.3f}\tTime: {:.3f}'.format(
                phase, epoch + 1, epochs, accs.avg, losses.avg, elapsed_time))
            epoch_loss[phase].append(losses.avg)
            #             epoch_bool_acc[phase].append(bool_accs.avg)
            epoch_acc[phase].append(accs.avg)
            #             epoch_recall[phase].append(recalls.avg)
            #             epoch_precision[phase].append(precisions.avg)

            if phase == 'val' and accs.avg > best_acc:
                best_acc = accs.avg

                best_epoch = epoch
                best_model_state = net.state_dict()
                preds = None
                targets = None
                bools = None
                #                 test_bool_accs=AverageMeter()
                test_accs = AverageMeter()
                y_true = None
                y_pred = None
                for l in tqdm(net.test_iter):
                    net.eval()
                    (input_seqs, input_seq_lens), ys = l
                    net.optimizer.zero_grad()  # clear the gradient
                    logits = net(input_seqs, input_seq_lens)

                    output = logits
                    l_n = logits.data.cpu().numpy()
                    nsample = input_seqs.size(0)

                    #                     bool_out=bool_out.view(bool_y.size(0))
                    #                     bool_acc=eval_(bool_out,bool_y.float(),binary=True)
                    acc = eval_(output, labels=ys.data.long())
                    _, predicted = torch.max(logits.cpu().data, 1)
                    test_accs.update(acc, nsample)
                    #                     test_bool_accs.update(bool_acc, nsample)
                    if y_true is None:
                        y_true = ys.data.cpu().numpy()
                        y_pred = l_n.argmax(axis=1)
                    else:
                        y_true = np.hstack([y_true, ys.data.cpu().numpy()])
                        y_pred = np.hstack([y_pred, l_n.argmax(axis=1)])

                print('[test]\tEpoch: {}/{}\tAcc: {:.2%}\tTime: {:.3f}'.format(
                    epoch + 1, epochs, test_accs.avg, elapsed_time))
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true, y_pred, labels=range(n_label))
                print(cm.shape)
                print_cm(cm, label_name)

    print('[Info] best valid acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch))
    torch.save(best_model_state, config.model_save_dir)
    print('Test Acc: {:.2%}\t'.format(
        test_accs.avg))

    plot = False
    if plot == True:
        for phase in ('train', 'val'):
            # plt.plot(range(len(epoch_loss[phase])), epoch_loss[phase], label=(phase + '_loss'))
            plt.plot(range(len(epoch_acc[phase])), epoch_acc[phase], label=(phase + '_F1'))
            plt.plot(range(len(epoch_bool_acc[phase])), epoch_bool_acc[phase], label=(phase + '_is_acc'))
            plt.plot(range(len(epoch_precision[phase])), epoch_precision[phase], label=(phase + '_is_acc'))
            plt.plot(range(len(epoch_recall[phase])), epoch_recall[phase], label=(phase + '_is_acc'))
            plt.plot(range(len(epoch_h_loss[phase])), epoch_h_loss[phase], label=(phase + '_is_acc'))

            plt.legend(prop={'size': 15})
            plt.savefig(log_dir + metric + "_multitask_" + 'res.png')
