# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from numpy import genfromtxt
import copy
import torch.nn.functional as F
import math

class Modelconfig():
    def __init__(self):
        self.rnn_features_inp_size = 5
        self.rnn_features_hid_size = 5
        self.rnn_features_layers = 1
        self.rnn_div_inp_size = 4
        self.rnn_div_hid_size = 4
        self.rnn_div_layers = 1
        self.rnn_weight_inp_size = 1
        self.rnn_weight_hid_size = 1
        self.rnn_weight_layers = 1
        self.rnn_diff_inp_size = 4
        self.rnn_diff_hid_size = 4
        self.rnn_diff_layers = 1
        self.rnn_final_inp_size = 1
        self.rnn_final_hid_size = 1
        self.rnn_final_layers = 1
        self.lin1_in= 18
        self.lin1_out = 50
        self.lin2_in = 50
        self.lin2_out = 1

class Model(nn.Module):
    ## ------------------------------MODEL VISUAL-------------------------------------------------
    ##               ▲--ScanScore&NormalDiff--> | Multiply |--MulHidden(1)-----> | Linear1|     | Linear2|    | RNN   |
    ##               |                          | Rnn?     |         ▲---------> | Layer  | --> | Layer  | -> | Layer | -> Final Output
    ##--Features(5)--x--x--------| Features |------HiddenFeatures(5)-x ▲-------> | 14x?   |     | ?x1    |    | Label |
    ##                  |        | RNN      |                          | ▲-----> |        |
    ##                  ▼ -AlgCurr(4)-->| Diff |------DiffHidden(4)----x |
    ## --MapAv(4)---------------------->| Rnn  |                         |
    ## --Divergences -(4)-▶ | DIV | ---------DIVHidden(4)----------------x
    ##                      | RNN |
    ##-----------------------------------------------------------------------------------------------
    def __init__(self, seq_len, batch_size,dev):
        super(Model, self).__init__()
        self.device = dev
        self.sequence_length = seq_len
        self.batch_size = batch_size
        self.config = Modelconfig();
#        self.rnn_final = nn.RNN(input_size=self.config.rnn_final_inp_size,
#                          hidden_size=self.config.rnn_final_hid_size,
#                          num_layers=self.config.rnn_final_layers,
#                          batch_first=True)
        # Try 1d convolution here
        self.Conv1d = nn.Conv1d(100, 1, 1).to(self.device)
        self.linear = torch.nn.Linear(in_features = 15, out_features=40,bias=True).to(self.device)
        self.linear2 =  torch.nn.Linear(in_features = 40, out_features=2,bias=True).to(self.device)
#        self.linear2 = torch.nn.Linear(in_features = 20, out_features=2,bias=True).to(self.device)
        #self.linear3 = torch.nn.Linear(in_features = 10, out_features=2,bias=True).to(self.device)
        # self.w_scan = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # Scan weight
        # self.w_pose = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # Pose diff weight
        # self.w_cov = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # Covariance weight
        # self.w_hood = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # likelihood 
        # self.w_adif = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # angle diff
        # self.w_sens = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # sensor div
        # self.w_odx = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # odom x div
        # self.w_odw = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # odometry w div
        # self.w_imw = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # imu w div
        # self.w_gkdt = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # kdtree diff
        # self.w_gcov = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # cov diff
        # self.w_glik = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # likelihood diff
        # self.w_gnor = torch.tensor(0.1, dtype=torch.float32, requires_grad=True) # normal diff
        # self.w_mul = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

    def initialize_hiddens(self):
        final_hidden = torch.ones(self.config.rnn_final_layers, self.batch_size, self.config.rnn_final_hid_size).to(self.device) * 0.5
        return final_hidden

    def forward(self, features, divergences, grid_avs ): #features_hidden=None, div_hidden=None, diff_hidden=None, multiply_hidden=None, final_hidden=None):
        final_hidden = self.initialize_hiddens();
        av_features = torch.cat((torch.stack((features[:,:,0], features[:,:,2]), -1) , torch.stack((features[:,:,3], features[:,:,4]), -1)), 2)
        diff_features = av_features - grid_avs
        epsilon = 0.01
        grid_avs = grid_avs + epsilon
        diffp_features = diff_features/grid_avs
        mul_feature = (features[:,:,0]*features[:,:,4] / 100.0).unsqueeze(-1)
        mul_av = (av_features[:,:,0]*av_features[:,:,3] / 100.0).unsqueeze(-1)
        mull_diff = mul_av - mul_feature
        comb1 = torch.cat((features, diff_features, divergences, mul_feature, mull_diff), 2).to(self.device)
        combConvolved = self.Conv1d(comb1)
        lin1_out = torch.sigmoid(self.linear(combConvolved)) # This is now in one hot encoding
        lin2_out = torch.sigmoid(self.linear2(lin1_out))
        return lin2_out, final_hidden


    def init_hidden_single(self, initial_vel,initial_yaw):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        #print("Initializing yaw at ", initial_yaw)
        return initial_vel.clone().detach().unsqueeze(0).unsqueeze(0), torch.ones(self.rnn2_n_layers, self.batch_size, self.rnn2_hidden_size)*initial_yaw.detach()
