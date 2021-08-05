import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.metrics.cluster import adjusted_rand_score
from test_model import AutoEncoder
from numpy import load
import numpy as np
import os
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from scipy import sparse
import configparser
import argparse
import sys
import distutils
import time
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class RandomDataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        self.data = data

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).cuda()

    def __len__(self):
        return self.len

def reduce_loss(loss,world_size):
    with torch.no_grad():
        dist.all_reduce(loss)
        loss /= world_size
        return loss.item()


def train(data,model):

    lr = 0.007
    decay_rate = 0.995
    batch_size = 256
    max_iter = 1000
    train_mode = True

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_world_size = 8
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda')

    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    torch.cuda.set_device(local_rank)

    torch.cuda.synchronize()
    start = time.time()
    #model = nn.DataParallel(model)
    #model = model.cuda()
    model = model.cuda(device_ids[0])
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids)
    
    '''
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    '''
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    lambda_epoch = lambda i_iter: world_size * lr * (decay_rate ** i_iter)
    scheduler = optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=lambda_epoch)
    model_path = '/home/ubuntu/uc2/science/CloudTest/model'
    para_path = '/home/ubuntu/uc2/science/CloudTest/para'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(para_path):
        os.mkdir(para_path)
 
    #adata_index_set = np.arange(len(data))
    dataset = RandomDataset(data)
    train_sampler = DistributedSampler(dataset)
    #train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    rand_loader = DataLoader(dataset=dataset,
             #batch_sampler=train_batch_sampler,
             sampler = train_sampler,
             batch_size = batch_size)
   
    loss_ = []
    if train_mode:
        logging.warning('model has been submit to cudas, training start!')
        with tqdm(total=max_iter) as pbar:
            for i_iter in range(max_iter):
                #print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, i_iter))
                pbar.update(1)
                loss_average, k = 0, 0
                train_sampler.set_epoch(i_iter+1)
                #while len(adata_index_set):
                #    batch_index = np.random.choice(adata_index_set, batch_size)
                #    adata_index_set = list(set(adata_index_set) - set(batch_index))
                #    input_adata = torch.tensor(data[batch_index]).cuda()
                for input_adata in rand_loader:
                    recon_adata, latent_space = model(input_adata)
                    loss = F.mse_loss(recon_adata, input_adata)
                    k += 1
                    loss_average += reduce_loss(loss,world_size)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                if rank == 0:
                    loss_.append(loss_average/k)
                #if i_iter % 5 == 0 and i_iter:
                    #print(i_iter)
                    # model_ = torch.nn.DataParallel(model)
                    # model_ = model_.module()
                    #print(model)
                    #torch.save(model.module.state_dict(), os.path.join(model_path,"model_{}.pt".format(str(i_iter))))
                    #torch.save(model.module.encoder_layers.state_dict(), os.path.join(para_path,"para_{}.pt".format(str(i_iter))))


    model.eval()
    torch.cuda.synchronize(device)
    end = time.time()
    if rank == 0:
        print(end-start)
        with open('/shared/loss_file_0805.txt', 'w') as f:
                for item in loss_:f.write("%s\n" % item)
    
