import torch
#import sequicity.config as seq_cfg
import os, sys
import numpy as np

class Config(object):
    # policy
    rule_policy = True
    sl_policy = not rule_policy

    # nlg
    nlg_template = False
    nlg_sample = False
    nlg_generation = True
    assert np.sum([nlg_template, nlg_sample, nlg_generation]) == 1




    # False-True-False-False ok
    # False-False-True-False error

    # True-True_False-False