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
from thesis_model import Model
from sklearn.metrics import confusion_matrix
import seaborn as sns


if(not torch.cuda.is_available()):
    print("Gpu not availiable exiting")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_grad_flow(named_parameters, ax3, ax4):
    ave_conv = []
    layers_conv = []
    ave_linear = []
    layers_linear = []
    ave_linear2 = []

    weight_list = ["scan", "pose", "covar", "lhood", "norm", "Dscan", "Dcov", "Dlhood", "Dnorm", "sensor", "odomx", "odomw", "imuw", "mull", "mull_diff"]

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and ("linear" in n):
            count_lin = 0
            count_lin2 = 0
            for item in p[0]:
                layers_linear.append(n)
                ave_linear.append(item)
                count_lin +=1
            for item in p[1]:
                layers_linear.append(n)
                ave_linear2.append(item)
                count_lin2 +=1

        elif (p.requires_grad) and ("bias" not in n):
            count = 0
            conv_mean = 0
            for item in p[0]:
                layers_conv.append("{}{}".format(n,count))
                ave_conv.append(item)
                count +=1
                conv_mean += item.detach().cpu()
            conv_mean = conv_mean / count
    mean = []
    for item in layers_conv:
        mean.append(conv_mean)
    ax3.plot(ave_conv, alpha=0.3, color="b")
    ax3.plot(mean, alpha=0.3, color="g")

    ax3.hlines(0, 0, len(ave_conv)+1, linewidth=1, color="k" )
#    ax3.xticks(range(0,len(ave_conv), 1), layers_conv, rotation="vertical")
    ax3.set_xlim([0, len(ave_conv)])
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Weights")
    ax3.set_title("Conv Weights")
    ax3.grid(True)

    ax4.plot(ave_linear, marker='o',alpha=0.3, color="b")
    ax4.plot(ave_linear2, marker='o',alpha=0.3, color="r")
    ax4.hlines(0, 0, len(ave_linear)+1, linewidth=1, color="k" )    
    ax4.set_xticks(np.arange(len(layers_linear)-1))
    ax4.set_xticklabels(labels=layers_linear, rotation=40, minor=False)
#    ax4.set_xticks(layers_linear, minor=True)
    ax4.set_xlim([0,len(ave_linear)-1])
    ax4.set_xlabel("Layers")
    ax4.set_ylabel("Weights")
    ax4.set_title("Linear Layer Weights")
    ax4.legend(['Localized', 'Lost'])
    ax4.grid(True)



def getConfusionElements(output, label):
    tp = 0; fp = 0; fn = 0; tn = 0;
    print("Output shape ",output.shape)
    print("Output is", output[0,0])
    for i in range(0,len(output)):
        if(output[i,0] == 1):
            #Positive
            if(label[i,0] == 1):
                tp += 1
            else:
                fp += 1
        else:
            if(label[i,0] == 1):
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def baseMetricsFromElements(tp,fp,fn,tn):
    precision = tp/(fp+tp);
    recall_lost = tp/(tp+fn)
    recall_localized = tn/(tn+fp)
    f1 = 2*precision*recall_lost / (precision + recall_lost)
    sensitivity = recall_lost
    specifity = recall_localized
    return precision, recall_lost, recall_localized, f1, sensitivity, specifity

def getMetrics(output, label):
    num_classes = 2;
    out_binary = output > 0.5
    print("Out binary shape {}, length {}".format(out_binary.shape, len(out_binary)))
    out_res = out_binary.reshape(len(out_binary), 1)
    compare = torch.arange(num_classes).reshape(1,num_classes)
    out_onehot = (torch.tensor(out_res) == compare).float()
    print("Out onehot \n", out_onehot)

    tp, fp, fn, tn = getConfusionElements(out_binary, label);
    print("Confusion is {}, {} \n              {}, {}".format(tp,fp,fn,tn))
    [precision, recall_lost, recall_localized, f1, sensitivity, specifity] = baseMetricsFromElements(tp,fp,fn,tn)
    print("Precision {}%, r_lost {}%, r_local {}%, f1 {}%".format(precision*100, recall_lost*100, recall_localized*100, f1*100))
    label_onehot = (torch.tensor(label).reshape(len(label),1) == compare).float()

    cm = confusion_matrix(out_onehot.numpy().argmax(axis=1), label_onehot.numpy().argmax(axis=1))

    plt.figure(2)
    plt.clf()
    ax=plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('True labels');ax.set_ylabel('Predicted labels'); 
    ax.set_title('Confusion Matrix', y=1.08); 
    ax.xaxis.set_ticklabels(['Localized', 'Lost']);
    ax.xaxis.set_label_position('top') 
    ax.yaxis.set_ticklabels(['Localized', 'Lost']);
    plt.yticks(rotation=0) 
    return 

def formatData(dataset):
    it = 13
    dataset = np.delete(dataset, 0, 0)
    labels = copy.deepcopy(dataset.transpose()[it+1][:]).transpose()
    #print("Labels shape {}".format(labels.shape))
    labels = F.one_hot(torch.tensor(labels).long(), num_classes=2)
    #print("Labels ONEHOT shape {}".format(labels.shape))
    #dataset = copy.deepcopy(dataset.transpose()[14:42][:]).transpose()
    features = copy.deepcopy(dataset.transpose()[it+3:it+7][:]).transpose()
    features = np.vstack([features.transpose(), copy.deepcopy(dataset.transpose()[it+10][:])]).transpose()
    divergences = copy.deepcopy(dataset.transpose()[it+13:it+17][:]).transpose()
    grid_avs = copy.deepcopy(dataset.transpose()[it+25:it+29][:]).transpose()
    return labels, dataset, divergences, grid_avs, features

def split_to_labels(dataset):
    transposed = copy.deepcopy(dataset.transpose());
    labels = copy.deepcopy(transposed[7:10][:])
    labels = labels.transpose()
    print("Shape is ", transposed.shape)
    transposed = np.delete(transposed, [7,8,9,10], 0);
    features = transposed.transpose();
    return features, labels

def form_windows(features, labels, sequence_length):
    # Pre allocation
    inp_size = features.shape[1]
    print("Windows Labels shape ", labels.shape)
    label_shape = labels.shape[1]
    wFeatures = np.zeros([features.__len__() - sequence_length + 1, sequence_length, inp_size]);
    wLabels = np.zeros([features.__len__() - sequence_length + 1, sequence_length, label_shape]);
    # wFeatures = 16, 5, 4
    # features =
    for j in range(0, features.__len__() - sequence_length + 1):
        temp = labels[j : j+sequence_length, :];
        wFeatures[j, 0 : sequence_length, :] = features[j : j+sequence_length, :]; # Assigns rows to w_features
        wLabels[j, 0:sequence_length, :] = labels[j : j+sequence_length, :];
    return torch.tensor(wFeatures, dtype=torch.float32), torch.tensor(wLabels, dtype=torch.float32)


def getTestFeature(feature_size, sequence_length):
    #Shape 1xsequence_lengthxno_features
    # Starting with ramp
    custom_input = torch.tensor(np.zeros([1, sequence_length-1, feature_size], dtype=float), dtype=torch.float64);

    for i in range(0, sequence_length-1):
        custom_input[0,i,7] = 1;
    return custom_input;

def formatLabelWithProbability(labels):
    prob_labels = copy.deepcopy(labels);
    curr_label = prob_labels[0]
    p_lost = 0.2
    p_local = 0.8
    if(curr_label == 1):
        p_lost = 0.7
        p_local = 0.3

    Zlost_at_lost = 0.8
    Zlocal_at_lost = 0.2
    Zlocal_at_local = 0.8
    Zlost_at_local = 0.2
    eps = 0.01
    prob_labels[0] = p_lost
    for i in range(1,len(labels)):
        if(labels[i] == 0):
            p_lost = Zlocal_at_lost * p_lost + eps
            p_local = Zlocal_at_local * p_local + eps
            norm = p_lost + p_local
            p_lost /= norm
            p_local /= norm
        else:
            p_lost = Zlost_at_lost * p_lost + eps
            p_local = Zlost_at_local * p_local + eps
            norm = p_lost + p_local
            p_lost /= norm
            p_local /= norm
        prob_labels[i] = p_lost
    return prob_labels

def loadEverything(path, files):
    first = True
    labels = np.zeros((1,1))
    for file in files:
        if not "csv" in file:
            continue
        print("Label cont shape {}".format(labels.shape))
        if(file=="onedata"):
            continue;
        filename = path+"/"+file
        dataset = genfromtxt(filename, delimiter=',')
        [tlabels, tdataset, tdivergences, tgrid_avs, tfeatures] = formatData(dataset)
        if(len(tlabels) < sequence_length):
            continue;
        if(first):
            [labels, dataset, divergences, grid_avs, features] = formatData(dataset)
            #prob_labels = formatLabelWithProbability(labels)
            #prob_labels = torch.tensor(prob_labels).unsqueeze(1)
            prob_labels = labels
            features, labels = form_windows(features, prob_labels, sequence_length);
            divergences, grid_avs = form_windows(divergences, grid_avs, sequence_length);
            first = False
            loaded = True
        else:
            [tlabels, tdataset, tdivergences, tgrid_avs, tfeatures] = formatData(dataset)
            #prob_labels = formatLabelWithProbability(tlabels)
            #prob_labels = torch.tensor(prob_labels).unsqueeze(1)
            prob_labels = tlabels
            tfeatures, tlabels = form_windows(tfeatures, prob_labels, sequence_length);
            tdivergences, tgrid_avs = form_windows(tdivergences, tgrid_avs, sequence_length);
            labels = torch.cat((tlabels, labels),0)
            divergences = torch.cat((tdivergences, divergences),0)
            features = torch.cat((tfeatures, features),0)
            grid_avs = torch.cat((tgrid_avs, grid_avs),0)
    feature_loader = torch.utils.data.DataLoader(
        features,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False
    )

    div_loader = torch.utils.data.DataLoader(
        divergences,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False
    )

    grid_avs_loader = torch.utils.data.DataLoader(
        grid_avs,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False
    )

    label_loader = torch.utils.data.DataLoader(
        labels,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False
    )
    return loaded, features, labels, divergences, grid_avs, feature_loader, div_loader, grid_avs_loader, label_loader


num_classes = 5
input_size = 8  # one-hot size
hidden_size = 3  # output from the RNN. 5 to directly predict one-hot
batch_size = 100   #
sequence_length = 100  # One by one
num_layers = 5  # one-layer RNN

rnn1_inp_size = 2
rnn1_hidden_size = 2
rnn1_n_layers = 1 # Like a 4th order filter

rnn2_inp_size = 1 # Just the angular velocity input
rnn2_hidden_size = 1 # Just output the yaw
rnn2_n_layers = 1 # yaw = rnn_hidden + deltat * rnn_inp

lin_layers = 3

#dataset2 = genfromtxt('/home/ulas/Desktop/milvus_python_ulas/data/odometry_calibration5_new.csv', delimiter=',')
#dataset3 = genfromtxt('/home/ulas/Desktop/milvus_python_ulas/data/odometry_calibration_test_inputs.csv', delimiter=',')
#dataset = genfromtxt('/home/ulas/Desktop/milvus_python_ulas/data/odometry_calibration_test_inputs.csv', delimiter=',') # test dataset
# print("Labels: \n ", labels);
# print("Divs: \n ", divergences);
# print("GridAvs: \n ", grid_avs);
# print("Features \n", features)

model = Model(sequence_length, batch_size, device).to(device)
# Forming windows here with sequence_length Inputs : [batch, seq, Input_size]
#                                           Labels: [batch, 1, Label_size]

# Load data here
path = "/home/ulas/milvus_python_ulas/data/csv/critically_lost/reformatted/legendary/combined_with_new/"
files = os.listdir(path)



do_training_ = True # Training parameters here
train_velocity_part_ = False
train_one_step_ = False
detach_hidden_ = False
load_state_dict_ = False
visualize_ = True
split_amount = 2
Learning_rate = 0.001 # 0.001 for velocity training


if(load_state_dict_):
    model.load_state_dict(torch.load("/home/ulas/milvus_python_ulas/pytorch/models/save/multi_files.txt"))

print(device)
print(model)

criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
steps = 19
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
# # Train the model
loss_container = np.array([1], dtype=float);
switch = 0;

fig = plt.figure(1)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


for epoch in range(2000):
    loaded, features, labels, divergences, grid_avs, feature_loader, div_loader, grid_avs_loader, label_loader = loadEverything(path, files)
    if(not loaded):
        continue;
    output_cont = np.zeros([labels.shape[0], 2]);
    label_cont = np.zeros([labels.shape[0], 2]);
    if(epoch%20 == 0):
         ax3.clear()
         ax4.clear()

    total_loss = 0
    for count, (feature, div, grid_avs, label) in enumerate(zip(feature_loader, div_loader, grid_avs_loader, label_loader)):
        [output, final_hidden] = model(feature, div, grid_avs)#, features_hidden, div_hidden, diff_hidden, multiply_hidden, final_hidden) # Move 1 step
        label = label.to(device)
        loss = criterion(output.squeeze(1), label[:,sequence_length-1, :]);
        output_cont[(count)*batch_size:(count+1)*batch_size,:] = copy.deepcopy(output[:,0,:].detach().view(output.size(0),-1).cpu().numpy())
        label_cont[(count)*batch_size:(count+1)*batch_size,:] = copy.deepcopy(label[:,sequence_length-1,:].detach().view(output.size(0),-1).cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        plot_grad_flow(model.named_parameters(),ax3, ax4)
        if(do_training_):
            optimizer.step()
            scheduler.step()
    print("Epoch {} {}".format(epoch, loss))
#        getMetrics(output_cont, label_cont)
    if(visualize_):
        ax1.clear()
        ax2.clear()
        size1 = output_cont[:, 0].shape[0]
        ax1.plot(label_cont[:, 0], 'r-', linewidth=1, label='Localized Labels')
        ax1.plot(output_cont[:, 0], 'c-', linewidth=3, label='Localized Prediction')
        ax1.legend(loc=1);
        ax1.set_title("Epoch {}".format(epoch+1), fontdict=None, loc='center')
        ax2.plot(label_cont[:, 1], 'r-', linewidth=1, label='Lost Labels')
        ax2.plot(output_cont[:, 1], 'c-', linewidth=3, label='Lost Prediction')
        ax2.legend(loc=1);
        ax2.set_title("Epoch {}".format(epoch+1), fontdict=None, loc='center')
        plt.pause(0.1)#(0.1)
    torch.save(model.state_dict(), "/home/ulas/milvus_python_ulas/pytorch/models/thesis_model_running{}.txt".format(epoch))
    fig.savefig("/home/ulas/milvus_python_ulas/pytorch/models/images/thesis_model_running_{}.png".format(epoch))

print("Learning finished!")
