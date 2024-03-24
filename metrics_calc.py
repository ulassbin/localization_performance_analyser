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
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

def getConfusionElements(output, label):
    tp = 0; fp = 0; fn = 0; tn = 0;
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
    recall_lost = tp/(tp+fn+1)
    recall_localized = tn/(tn+fp+1)
    f1 = 2*precision*recall_lost / (precision + recall_lost+1)
    sensitivity = recall_lost
    specifity = recall_localized
    return precision, recall_lost, recall_localized, f1, sensitivity, specifity

def plotROC(output, label):
    plt.figure(4)
    plt.clf()
    out_arr = np.asarray(output.flatten())
    labl_arr = np.asarray(label.astype(int).flatten())
    fpr, tpr, threshold = metrics.roc_curve(labl_arr, out_arr)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1],'r--')
    return
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
    plotROC(output, label)
    return 


