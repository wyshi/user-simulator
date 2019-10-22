import os
import sys
import pickle as pkl
import numpy as np
sys.path.append("/home/wyshi/simulator")
log_temp_dir = "/home/xweiwang/test/systest/dialog{}/dialogs_pkl/"

len_list = []

import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

for i in range(1, 8):
    len_list_tmp = []
    for filename in os.listdir(log_temp_dir.format(i)):
        if (not filename.endswith('incomplete.pkl')) and (filename.endswith("pkl")):
            print(log_temp_dir.format(i)+filename)
            with open(log_temp_dir.format(i)+filename, "rb") as fh:
                c = pkl.load(fh)
                len_list_tmp.append(len(c['dialog']))
    len_list.append(len_list_tmp)

for i in range(7):
    m, ci = mean_confidence_interval(len_list[i])
    print("{}: mean, {}; ci, ({})".format(i+1, m, ci))




