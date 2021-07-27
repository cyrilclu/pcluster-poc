#!/usr/bin/env python
import logging
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool as ThreadPool
import multiprocessing as mp
import seaborn as sns
import os  
from tqdm import tqdm
import argparse

os.environ['PYTHONHASHSEED'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.cluster import adjusted_rand_score
from test_model import AutoEncoder
from numpy import load
from test_train import train
import pandas as pd
import sys
import distutils
from distutils import util
import pickle
import torch.distributed as dist


def main():

    with open('/home/ubuntu/uc2/science/CloudTest/Dataset_650.pkl', 'rb') as f: adata_X = pickle.load(f)
    n_gene = adata_X.shape[1]
    #print(n_gene)
    dim_arg = [500,128,64,16]
    dim = [n_gene] + dim_arg
    dim_lis = dim + dim[::-1][1:]

    model_dim = '_'.join([str(i) for i in dim_lis])
    model = AutoEncoder(dim)

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print ("[{}] Initializing process group with: {}".format(os.getpid(),env_dict))

    dist.init_process_group(backend="nccl")

    print("[{}] world_size = {}, rank = {}, backend={}".format(os.getpid(),dist.get_world_size(),dist.get_rank(),dist.get_backend()))

    train(adata_X,model)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
    #print (os.path.join(os.path.dirname(torch.__file__), 'distributed', 'launch.py'))


