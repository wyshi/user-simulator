from __future__ import print_function
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dir', help="data directory", default="../data")
parser.add_argument('--max-len', help="max sequence length", default=10)
args = parser.parse_args()

import json
import os
import pandas as pd
def generate_seq(group):
    actions=[]
    for index,row in group.iterrows():
        actions.append(row.label)
    
    return actions
def generate_dataset(root, name,df_tgt,df_src,start=0,end=0,src=1):
    '''
    src : =1 means EE as input sequence
            =0 means ER as input sequence
    '''
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)
    src_path = os.path.join(path, 'src.txt')
    tgt_path = os.path.join(path, 'tgt.txt')
#     dialog_dict_path='../../data/dict.json'
#     with open(dialog_dict_path) as json_data:
#         d_dict = json.load(json_data)
    with open(src_path, 'w') as src_out, open(tgt_path, 'w') as tgt_out:
        for i in range(start,end):
            tgt_tmp=df_tgt[df_tgt['B1']==i]
            src_tmp=df_src[df_src['B1']==i]
            if src_tmp.shape[0]==0 or tgt_tmp.shape[0]==0:
                print("Lack",i)
#                 print(tgt_tmp.B2.iloc[0])
                continue
            assert tgt_tmp.Turn.max()>=8
            assert src_tmp.Turn.max()>=8
            if  tgt_tmp.Turn.max()>= src_tmp.Turn.max():  #ER speak first
                first=1-src 
            else:  #EE speak first
                first=src

            tgt_list=tgt_tmp.groupby('Turn').apply(generate_seq)
            src_list=src_tmp.groupby('Turn').apply(generate_seq)
            if first== src:
                for i in range(min(src_list.shape[0],tgt_list.shape[0])):
                    
                    src_out.write(" ".join(str(a) for a in src_list[i]) + "\n")
                    tgt_out.write(" ".join(str(a) for a in tgt_list[i]) + "\n")
            else: 
                for i in range(min(src_list.shape[0],tgt_list.shape[0]-1)):
                    src_out.write(" ".join(str(a) for a in src_list[i]) + "\n")
                    tgt_out.write(" ".join(str(a) for a in tgt_list[i+1]) + "\n")


    #         print(tgt_list)
    #         print(src_list)

    #         input()


if __name__ == '__main__':
    data_dir = args.dir

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    toy_dir = os.path.join(data_dir, 'EE_data')
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    df_er=pd.read_csv('../../data/persuader/er_sum.csv',index_col=[0])
    df_ee=pd.read_csv('../../data/persuadee/ee_sum.csv',index_col=[0])
    generate_dataset(toy_dir, 'train', df_ee,df_er,end=200,src=0)       
    generate_dataset(toy_dir, 'dev',  df_ee,df_er,start=201,end=240,src=0)  
    generate_dataset(toy_dir, 'test',  df_ee,df_er,start=241,end=298,src=0)

# def generate_dataset(root, name, size):
#     path = os.path.join(root, name)
#     if not os.path.exists(path):
#         os.mkdir(path)

#     # generate data file
#     src_path = os.path.join(path, 'src.txt')
#     tgt_path = os.path.join(path, 'tgt.txt')
#     with open(src_path, 'w') as src_out, open(tgt_path, 'w') as tgt_out:
#         for _ in range(size):
#             length = random.randint(1, args.max_len)
#             seq = []
#             for _ in range(length):
#                 seq.append(str(random.randint(0, 9)))
#             src_out.write(" ".join(seq) + "\n")
#             tgt_out.write(" ".join(reversed(seq)) + "\n")


# if __name__ == '__main__':
#     data_dir = args.dir
#     if not os.path.exists(data_dir):
#         os.mkdir(data_dir)

#     toy_dir = os.path.join(data_dir, 'toy_reverse')
#     if not os.path.exists(toy_dir):
#         os.mkdir(toy_dir)

#     generate_dataset(toy_dir, 'train', 10000)
#     generate_dataset(toy_dir, 'dev', 1000)
#     generate_dataset(toy_dir, 'test', 1000)
