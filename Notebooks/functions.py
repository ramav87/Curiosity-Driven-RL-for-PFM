import gc
import pyUSID as usid
import gym
from gym import spaces
import numpy as np
#import pygame



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

start = [25, 25]
initialize = 30
num_points = 200
image_patch = 9
image_size = 49
y_dim = 50
ldim_i = 2
ldim_a = 3
lbda = 2
norm_pola = np.load('../Data Files/norm_polarization.npy')
image, spectra = norm_pola[:,:,25], norm_pola[:,:,49:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feature_extractor(model, X):
    return(model.encoder(torch.tensor(X).reshape(X.shape[0], 1, image_patch, image_patch)))

def get_image_patch(image, pos, image_patch):
    return(image[pos[0]-int((image_patch-1)/2): pos[0] + int((image_patch+1)/2), pos[1]-int((image_patch-1)/2): pos[1] + int((image_patch+1)/2)])

def err_train(model, X, reward, criterion, optimizer, autoencoder):
    
    data = feature_extractor(autoencoder, np.array(X)).float().to(device = device)
    targets = torch.tensor(reward).float().to(device = device)

    scores = model.forward(data)

    loss = criterion(scores, targets.reshape(targets.shape[0], 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def model_train(model, X, y, criterion, optimizer):
    model.train()
    scores = model(torch.tensor(X).to(device).reshape([X.shape[0], 1, X.shape[1], X.shape[2]]))
    targets = torch.tensor(np.array(y)).to(device)
    loss = criterion(scores, targets.reshape([targets.shape[0], 1, targets.shape[1]]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()

class im2spec(nn.Module):
    """
    Encoder (2D) - decoder (1D) type model for generating spectra from image
    """
    def __init__(self,
                 feature_size: Tuple[int, int],
                 target_size: int,
                 latent_dim: int,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64) -> None:
        super(im2spec, self).__init__()
        self.n, self.m = feature_size
        self.ts = target_size
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec
        # Encoder params
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True, dropout_ = 0.5)
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        
        self.dec_fc1 = nn.Linear(latent_dim, self.ts //4 )
        self.dec_fc2 = nn.Linear(self.ts // 4, self.ts //4 * 2 )
        self.dec_fc3 = nn.Linear(self.ts //4 * 2, self.ts //4 * 3 )
        self.dec_fc4 = nn.Linear(self.ts //4 * 3, self.ts)
        self.dec_fc5 = nn.Linear(self.ts, self.ts)
        self.dec_fc6 = nn.Linear(self.ts, self.ts)
        
        self.dec_fc = nn.Linear(latent_dim, self.d_filt*self.ts)
        self.dec_atrous = dilated_block(
            ndim=1, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv = conv_block(
            ndim=1, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_out = nn.Conv1d(1, 1, 1)
        '''
        self.dec_fc = nn.Linear(latent_dim, self.d_filt*self.ts)
        self.dec_atrous = dilated_block(
            ndim=1, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv = conv_block(
            ndim=1, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_out = nn.Conv1d(1, 1, 1)
        '''
    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)

    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 1D signal from the embedded features
        """
        
        x = F.relu(self.dec_fc1(encoded))
        x = F.relu(self.dec_fc2(x))
        x = F.relu(self.dec_fc3(x))
        x = F.relu(self.dec_fc4(x))
        x = F.relu(self.dec_fc5(x))
        
        return self.dec_fc6(x).reshape(-1, 1, self.ts)
        '''
        x = self.dec_fc(encoded)
        x = x.reshape(-1, self.d_filt, self.ts)
        x = self.dec_atrous(x)
        x = self.dec_conv(x)
        return self.dec_out(x)
        '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        encoded = self.encoder(x)
        return self.decoder(encoded)

class im2im(nn.Module):
    def __init__(self,
                 feature_size: Tuple[int, int],
                 latent_dim: int = 10,
                 nb_filters_enc: int = 64,
                 nb_filters_dec: int = 64) -> None:
        super(im2im, self).__init__()
        self.n, self.m = feature_size
        self.e_filt = nb_filters_enc
        self.d_filt = nb_filters_dec
        # Encoder params
        self.enc_conv = conv_block(
            ndim=2, nb_layers=3,
            input_channels=1, output_channels=self.e_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.enc_fc = nn.Linear(self.e_filt * self.n * self.m, latent_dim)
        # Decoder params
        self.dec_fc = nn.Linear(latent_dim, self.d_filt * (self.n//4) * (self.n//4))
        self.dec_conv_1 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv_2 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=self.d_filt,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_atrous = dilated_block(
            ndim=2, input_channels=self.d_filt, output_channels=self.d_filt,
            dilation_values=[1, 2, 3, 4], padding_values=[1, 2, 3, 4],
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_conv_3 = conv_block(
            ndim=2, nb_layers=1,
            input_channels=self.d_filt, output_channels=1,
            lrelu_a=0.1, use_batchnorm=True)
        self.dec_out = nn.Conv2d(1, 1, 1)
    
    def encoder(self, features: torch.Tensor) -> torch.Tensor:
        """
        The encoder embeddes the input image into a latent vector
        """
        x = self.enc_conv(features)
        x = x.reshape(-1, self.e_filt * self.m * self.n)
        return self.enc_fc(x)
    
    def decoder(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        The decoder generates 2D image from the embedded features
        """
        x = self.dec_fc(encoded)
        x = x.reshape(-1, self.d_filt, self.n//4, self.n//4)
        x = self.dec_conv_1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.dec_conv_2(x)
        x = F.interpolate(x, scale_factor=self.n/(2 * (self.n//4)), mode="nearest")
        x = self.dec_atrous(x)
        x = self.dec_conv_3(x)

        return self.dec_out(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model"""
        encoded = self.encoder(x)
        return self.decoder(encoded)
    
class rewards_model(nn.Module):

    def __init__(self, n_observations):
        super(rewards_model, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(128, 1)
        
    def forward(self, x, dropout = True):
        
        x = F.relu(self.layer1(x))
        if dropout:
            x = self.dropout(x)
        
        return self.layer2(x)

    # def __init__(self, input_size, hidden_size, num_layers = 1):
    #     super(rewards_model, self).__init__()
        
    #     self.num_layers = num_layers #number of layers
    #     self.input_size = input_size #input size
    #     self.hidden_size = hidden_size #hidden state
        

    #     self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
    #                       num_layers=num_layers, batch_first=True) #lstm
    #     self.fc = nn.Linear(hidden_size, 1) #fully connected last layer

    
    # def forward(self,x):
    #     h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
    #     c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
    #     # Propagate input through LSTM
    #     output, (hn, cn) = self.lstm(Variable(x), (h_0, c_0)) #lstm with input, hidden, and internal state
    #     hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
       
    #     return self.fc(hn)
    
        
def find_max_in_keys(dictionary):
    max_first = float('-inf')  # Initialize to negative infinity to handle all positive numbers
    max_second = float('-inf')
    
    for key in dictionary.keys():
        try:
            first, second = map(int, key.split(':'))  # Split key by ':' and convert to integers
            if first > max_first:
                max_first = first
            if second > max_second:
                max_second = second
        except:
            pass
    
    return max_first, max_second


class environment(gym.Env):
    def __init__(self, image, spectra, start = [50, 50], image_patch = 5, image_size = 100, y_dim = 64):
        super(environment, self).__init__()
        self.num_not_measure = 0
        self.num_measure = 0
        self.image_patch = image_patch
        self.image_size = image_size
        self.radius = int((image_patch - 1)/2)+1
        self.image = image
        self.spectra = spectra
        self.y_dim = y_dim
        #self.color = 255/(image.max() - image.min()) * image - 255/(image.max() - image.min()) * image.min()
        self.num_rows = image_size
        self.num_columns = image_size
        self.all_X = []
        self.X = []
        self.y = []
        self.all_X.append(get_image_patch(self.image, start, self.image_patch))
        self.X.append((get_image_patch(self.image, start, self.image_patch)))
        self.y.append(spectra[start[0], start[1]])
        self.seen = np.zeros([self.image_size, self.image_size])
        self.pos_X = []
        self.pos_X.append(start)
        self.pos = [start[0], start[1]]
        self.measured = np.zeros([self.image_size, self.image_size])
        self.display = np.zeros([self.image_size, self.image_size])
        self.measured[start[0], start[1]] = 1
        
        for i in range(self.radius):
                for j in range(self.radius):
                    self.seen[start[0]+i, start[1]+j] = 1
                    self.seen[start[0]-i, start[1]+j] = 1
                    self.seen[start[0]+i, start[1]-j] = 1
                    self.seen[start[0]-i, start[1]-j] = 1
        
        
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_columns)))
        
        #pygame.init()
        #self.cell_size = 8
        #self.screen = pygame.display.set_mode((self.num_columns * self.cell_size, self.num_rows * self.cell_size))
    
    def update_pos(self):
        self.pos[0] = random.randint(self.radius-1, self.image_size - self.radius)
        self.pos[1] = random.randint(self.radius-1, self.image_size - self.radius)
        while self.measured[self.pos[0], self.pos[1]] == 1:
            self.pos[0] = random.randint(self.radius-1, self.image_size - self.radius)
            self.pos[1] = random.randint(self.radius-1, self.image_size - self.radius)
            
        self.all_X.append(get_image_patch(self.image, self.pos, self.image_patch))
    
    def step(self, action, display, num_epochs = 10):
            
        self.num_measure += 1
        self.pos = action
        ind = action
        self.measured[ind[0], ind[1]] = 1
        if display:
            self.display[ind[0], ind[1]] = 1
        for i in range(self.radius):
            for j in range(self.radius):
                self.seen[ind[0]+i, ind[1]+j] = 1
                self.seen[ind[0]-i, ind[1]+j] = 1
                self.seen[ind[0]+i, ind[1]-j] = 1
                self.seen[ind[0]-i, ind[1]-j] = 1
        self.X.append(get_image_patch(self.image, self.pos, self.image_patch))
        self.y.append(spectra[ind[0], ind[1]])
        
        self.pos_X.append([ind[0], ind[1]])

    
            
        
    
    def state(self):

        state = list(self.all_X[-1].reshape(self.image_patch**2))

        return(state)
    
    #def render(self):

        #self.screen.fill((255, 255, 255)) 
        
        #for row in range(self.num_rows):
            #for col in range(self.num_columns):
                #cell_left = col * self.cell_size
                #cell_top = row * self.cell_size
                
                #pygame.draw.rect(self.screen, (0, 0, self.color[row, col]), (cell_left, cell_top, self.cell_size, self.cell_size))
                
                #if self.display[row, col] == 1:
                    
                    #pygame.draw.rect(self.screen, (255, 0, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
                
                #if [row, col] == self.pos:
                    
                    #pygame.draw.rect(self.screen, (0, 255, 0), (cell_left, cell_top, self.cell_size, self.cell_size))
        
                

        #pygame.display.update()

            
    def reset(self, model, start = [50, 50]):
        self.num_measure = 0
        self.pos_X = [[start[0], start[1]]]
        self.all_X = []
        self.X = []
        self.y = []
        self.all_X.append(get_image_patch(self.image, start, self.image_patch))
        self.X.append((get_image_patch(self.image, start, self.image_patch)))
        self.y.append(spectra[start[0], start[1]])
        self.seen = np.zeros([self.image_size, self.image_size])
        self.pos = start
        self.measured = np.zeros([100, 100])
        self.measured[start[0], start[1]] = 1
        
        for i in range(self.radius):
                for j in range(self.radius):
                    self.seen[start[0]+i, start[1]+j] = 1
                    self.seen[start[0]-i, start[1]+j] = 1
                    self.seen[start[0]+i, start[1]-j] = 1
                    self.seen[start[0]-i, start[1]-j] = 1
        
        
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_columns)))

        #pygame.init()
        #self.cell_size = 8
        #self.screen = pygame.display.set_mode((self.num_columns * self.cell_size, self.num_rows * self.cell_size))
    