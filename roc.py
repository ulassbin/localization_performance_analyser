import sklearn.metrics as metrics
from thesis_model import Model
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from numpy import genfromtxt
import copy

def formatData(dataset):
    it = 13
    dataset = np.delete(dataset, 0, 0)
    labels = copy.deepcopy(dataset.transpose()[it+1][:]).transpose()
    #dataset = copy.deepcopy(dataset.transpose()[14:42][:]).transpose()
    features = copy.deepcopy(dataset.transpose()[it+3:it+7][:]).transpose()
    features = np.vstack([features.transpose(), copy.deepcopy(dataset.transpose()[it+10][:])]).transpose()
    divergences = copy.deepcopy(dataset.transpose()[it+13:it+17][:]).transpose()
    grid_avs = copy.deepcopy(dataset.transpose()[it+25:it+29][:]).transpose()
    return labels, dataset, divergences, grid_avs, features

def form_windows(features, labels, sequence_length):
    # Pre allocation
    inp_size = features.shape[1]
    label_shape = labels.shape[1]
    wFeatures = np.zeros([features.__len__() - sequence_length + 1, sequence_length, inp_size]);
    wLabels = np.zeros([features.__len__() - sequence_length + 1, sequence_length, label_shape]);
    # wFeatures = 16, 5, 4
    # features =
    for j in range(0, features.__len__() - sequence_length + 1):
        temp = labels[j : j+sequence_length, :];
        wFeatures[j, 0 : sequence_length, :] = features[j : j+sequence_length, :]; # Assigns rows to w_features
        wLabels[j, 0:sequence_length, :] = labels[j : j+sequence_length, :];
    return wFeatures, wLabels

def loadEverything(filename):
    dataset = genfromtxt(filename, delimiter=',')
    [labels, dataset, divergences, grid_avs, features] = formatData(dataset)
    loaded = False
    print("Labels shape ", labels.shape)
    if(labels.shape[0] < sequence_length):
        loaded = False
    else:
        loaded = True
        labels = torch.tensor(labels).unsqueeze(1);
        print("-------------------------")
        print("Loading {}".format(filename))
        print("Features {}, labels {}".format(features.shape, labels.shape))
        print("Div {}, gridAvs {}".format(divergences.shape, grid_avs.shape))
        features, labels = form_windows(features, labels, sequence_length);
        divergences, grid_avs = form_windows(divergences, grid_avs, sequence_length);

        features = torch.tensor(features).float().to(device);
        labels = torch.tensor(labels).float().to(device);
        divergences = torch.tensor(divergences).float().to(device);
        grid_avs = torch.tensor(grid_avs).float().to(device);
        print("Features {}, labels {}".format(features.shape, labels.shape))
        print("Div {}, gridAvs {}".format(divergences.shape, grid_avs.shape))
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





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_container = np.array([1], dtype=float);
switch = 0;

num_classes = 5
input_size = 8  # one-hot size
hidden_size = 3  # output from the RNN. 5 to directly predict one-hot
batch_size = 30   #
sequence_length = 100  # One by one
num_layers = 5  # one-layer RNN

rnn1_inp_size = 2
rnn1_hidden_size = 2
rnn1_n_layers = 1 # Like a 4th order filter

rnn2_inp_size = 1 # Just the angular velocity input
rnn2_hidden_size = 1 # Just output the yaw
rnn2_n_layers = 1 # yaw = rnn_hidden + deltat * rnn_inp

lin_layers = 3
model = Model(sequence_length, batch_size, device).to(device)
model.load_state_dict(torch.load("/home/ulas/milvus_python_ulas/pytorch/models/thesis_model_running.txt"))
# Load data here
path = "/home/ulas/milvus_python_ulas/data/csv/long/onedata"
files = os.listdir(path)

fig, axs = plt.subplots(2)

for file in files:
    print("File {}".format(file))
    loaded, features, labels, divergences, grid_avs, feature_loader, div_loader, grid_avs_loader, label_loader = loadEverything(path+"/"+file)
    if(not loaded):
        continue;
    output_cont = np.zeros([labels.shape[0], 1]);
    label_cont = np.zeros([labels.shape[0], 1]);

    total_loss = 0
    for count, (feature, div, grid_avs, label) in enumerate(zip(feature_loader, div_loader, grid_avs_loader, label_loader)):
        feature = feature.to(device);
        div = div.to(device);
        grid_avs = grid_avs.to(device);
        label = label[:,sequence_length-1,:].to(device); # Just get the latest element of the label
        [output, features_hidden, div_hidden, diff_hidden, multiply_hidden] = model(feature, div, grid_avs)#, features_hidden, div_hidden, diff_hidden, multiply_hidden, final_hidden) # Move 1 step
        output_cont[(count)*batch_size:(count+1)*batch_size,:] = copy.deepcopy(output.detach().view(output.size(0),-1).cpu().numpy())
        label_cont[(count)*batch_size:(count+1)*batch_size,:] = copy.deepcopy(label.detach().view(output.size(0),-1).cpu().numpy())

    print("Sizes {}, {}".format(output_cont.shape, label_cont.shape))
    
    # calculate the fpr and tpr for all thresholds of the classification
    out_arr = np.asarray(output_cont.flatten())
    labl_arr = np.asarray(label_cont.flatten())
    print("Sizes2 {}, {}".format(out_arr.shape, labl_arr.shape))
    
    fpr, tpr, threshold = metrics.roc_curve(labl_arr, out_arr)
    print("Threshold is ", threshold)
    roc_auc = metrics.auc(fpr, tpr)
    axs[0].clear()
    axs[1].clear()
    axs[1].plot(label_cont[:, 0], 'r-', linewidth=1, label='Labels')
    axs[1].plot(output_cont[:, 0], 'c-', linewidth=2, label='Model Prediction')
    axs[1].legend();
    axs[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    axs[0].legend(loc = 'lower right')
    axs[0].set(xlabel="False Positive Rate",ylabel="True Positive Rate")
    axs[0].plot([0, 1], [0, 1],'r--')
    # method I: plt
    plt.show()