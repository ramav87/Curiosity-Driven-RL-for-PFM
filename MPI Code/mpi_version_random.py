import gc
import pyUSID as usid
import gym
from gym import spaces
import numpy as np
from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import scipy
import h5py
import matplotlib.pyplot as plt
#import pyUSID as usid
import sidpy as sid

import os
from copy import deepcopy as dc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#from torch_kmeans import KMeans

from im2spec.models import conv_block, dilated_block
from im2spec.utils import create_training_set, predict, encode, decode
from im2spec.train_utils import trainer

import random
from collections import namedtuple, deque
from typing import List, Tuple
import math
from functions import *

norm_pola = np.load('norm_polarization.npy')

start = [25, 25]
initialize = 30
num_points = 35
image_patch = 9
image_size = 49
y_dim = 50
image, spectra = norm_pola[:,:,25], norm_pola[:,:,49:]
ldim_i = 2
ldim_a = 3
lbda = 2
num_trials = 5

radius = int((image_patch - 1)/2)
pos_X = []
X = []
y = []
for i in range(radius, image_size - radius):
    for j in range(radius, image_size - radius):
        pos_X.append([i, j])
        ind = pos_X[-1]
        X.append( get_image_patch(image, ind, image_patch))
        y.append(spectra[ind[0], ind[1]])
X = np.array(X)
y = np.array(y)
X = X.reshape([X.shape[0], 1, image_patch, image_patch])
y = y.reshape([y.shape[0], 1, y_dim])

X_tensor = torch.tensor(X).to(device)
y_tensor = torch.tensor(y).to(device)

autoencoder = im2im((image_patch, image_patch), ldim_a).to(device)
autoencoder = trainer(autoencoder, X, X, X, X, num_epochs=3, savename="im2spec_lv{}".format(rank)).run()

def run_agent_random(seed=None, device="cpu"):

    rand_loss_list = []
    rand_max_loss_list = []
    model = im2spec((image_patch, image_patch), y_dim, latent_dim = ldim_i).to(device)
    model_criterion = torch.nn.MSELoss()
    model_optimizer = optim.AdamW(model.parameters(), lr=0.001)
    model.eval()
    
    error_predictor = rewards_model(ldim_a).to(device)
    err_criterion = torch.nn.MSELoss()
    err_optimizer = optim.Adam(error_predictor.parameters(), lr=0.01)
    env = environment(image, spectra, start = start, image_patch = image_patch, image_size = image_size, y_dim = y_dim)
    featurizer = autoencoder
    im_feat = feature_extractor(featurizer, X)
    im_feat = im_feat.reshape([im_feat.shape[0], 1, im_feat.shape[1]]).to(device)
    action = start
    reward = [0]
    e_list = []
    max_e_list = []
    ep_list = []
    eu_list = []
    dist_reward_list = []
    lat_reward_list = []
    loss = []
    reward_list = []
    initialize = 30
    num_points = 200
    
    for i in range(initialize):
     
        while env.measured[action[0], action[1]] == 1:
                action[0] = random.randint(radius, image_size - radius -1)
                action[1] = random.randint(radius, image_size - radius -1)
        
        env.step(action, False)

    for i in range(5*initialize):
        model_train(model, np.array(env.X), np.array(env.y), model_criterion, model_optimizer)

        y_pred = np.array(model(X_tensor.to(device)).cpu().detach())
        err = (((y_pred - y)**2).sum(axis = 2)/y_dim).reshape(image_size - 2*radius, image_size - 2*radius)
       
        loss.append(err.sum())


    pred = model(torch.tensor(np.array(env.X).reshape([len(env.X), 1, image_patch, image_patch])).to(device)).cpu()
    gt = torch.tensor(np.array(env.y).reshape([len(env.y), 1, y_dim]))
    err_i = (((gt - pred)**2).sum(axis = 2)/y_dim).reshape(gt.shape[0])
    for i in range(2):
        model_train(model, np.array(env.X), np.array(env.y), model_criterion, model_optimizer)

    pred = model(torch.tensor(np.array(env.X).reshape([len(env.X), 1, image_patch, image_patch])).to(device)).cpu()
    err_f = (((gt - pred)**2).sum(axis = 2)/y_dim).reshape(gt.shape[0])
    
    reward = (err_i + err_f).detach()

    reward_list.append(reward)

    reward = reward / reward.mean()
    
    for i in range(2*initialize):
        err_train(error_predictor, np.array(env.X), np.array(reward), err_criterion, err_optimizer, featurizer)

    err_mean = np.zeros([image_size - 2*radius, image_size - 2*radius])
    err_std = np.zeros([image_size - 2*radius, image_size - 2*radius])
    dist_reward = np.zeros([image_size, image_size])
    lat_reward = np.zeros([image_size - 2*radius, image_size - 2*radius])
    
    if rank==0:
        print("Initialization complete")
        
    while env.num_measure < num_points:
        ep_list.append(err_mean)
        eu_list.append(err_std)
        dist_reward_list.append(dist_reward)
        while env.measured[action[0], action[1]] == 1:
            action[0] = random.randint(radius, image_size - radius -1)
            action[1] = random.randint(radius, image_size - radius -1)
       
        env.step(action, True)
        
        pred = model(torch.tensor(np.array(env.X).reshape([len(env.X), 1, image_patch, image_patch])).to(device)).cpu()
        gt = torch.tensor(np.array(env.y).reshape([len(env.y), 1, y_dim]))
        err_i = (((gt - pred)**2).sum(axis = 2)/y_dim).reshape(gt.shape[0])
        
        for i in range(2):
            model_train(model, np.array(env.X), np.array(env.y), model_criterion, model_optimizer)
        
        y_pred = np.array(model(X_tensor.to(device)).detach().cpu())
        err = (((y_pred - y)**2).sum(axis = 2)/y_dim).reshape(image_size - 2*radius, image_size - 2*radius)
        e_list.append(err)
        max_e_list.append(np.sort(e_list[-1].reshape(-1))[-10:].mean())
       
        loss.append(err.sum())
        pred = model(torch.tensor(np.array(env.X).reshape([len(env.X), 1, image_patch, image_patch])).to(device)).cpu()
        err_f = (((gt - pred)**2).sum(axis = 2)/y_dim).reshape(gt.shape[0])
        
        reward = (err_i + err_f).detach()
        reward_list.append(reward)
        reward = reward/reward.mean()
    
    rand_loss_list.append(min(loss))
    rand_max_loss_list.append(min(max_e_list))
     
    return [rand_loss_list, rand_max_loss_list]

if __name__ == "__main__":
    print('on rank {}'.format(rank))
    num_trials = 5
    for k in range(num_trials):
        loss_list, max_loss_list = run_agent_random()
        fname = 'results_from_rank_' + str(rank) +'_k=' + str(k) +'.npy'
        path = 'Results/Random/'
        file_path = os.path.join(path, fname)
        np.save(file_path, np.array([loss_list, max_loss_list]))
        print('completed trial {} of {}, rank {}'.format(k,num_trials, rank))
