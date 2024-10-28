import numpy as np
import os
import sys
import time
import json
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.loader import DataLoader
import scipy.io
from scipy.ndimage.interpolation import shift
from copy import copy

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Directories
CLASSES_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
cwd = DIR_EXPERIMENT
# print(os.path.abspath(__file__))
# print(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(CLASSES_DIR))

# Classes and Functions
from Classes.dataset import GraphDataset
from Classes.model import Net
from Classes.train_validate_fun import *
from Classes.plotting import plot_loss_acc_epochs_hyperparam



# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + '/../../../../../Dataset/'
name_dataset = 'bounding_boxes_noise.mat'
raw_data = scipy.io.loadmat(dir_dataset + name_dataset)
number_vehicles = 20
number_instants = 1000
raw_data = raw_data['new_dataset'][:,:]

# import vehicle positions
name_dataset_actors = 'actor_data.mat'
raw_data_vehicles = scipy.io.loadmat(dir_dataset + name_dataset_actors)
raw_data_vehicles = raw_data_vehicles['vehicles_pos'][0,0]
vehicles_pos = np.array([vehicles for vehicles in raw_data_vehicles])
# vehicles_pos[vehicle, instant]
vehicles_pos_xy = vehicles_pos[:,:,:2]
vehicles_pos_xy = {i:vehicles_pos_xy[i] for i in range(vehicles_pos_xy.shape[0])}

connectivity_matrix_dict = []
for instant in range(number_instants):
    connectivity_matrix_dict.append([])
    i = 0
    for vehicle in range(number_vehicles):
        connectivity_matrix_dict[instant].append({})
        for name in raw_data[vehicle][instant][0][0][1][0]:
            connectivity_matrix_dict[instant][vehicle][name[0]] = i
            i += 1


dataset = GraphDataset(root=dir_dataset, raw_data=raw_data, dir_dataset=dir_dataset, number_instants=number_instants, number_vehicles=number_vehicles, connectivity_matrix_dict=connectivity_matrix_dict)
dataset = dataset.shuffle()
train_dataset = dataset[:int(0.7*len(dataset))]
val_dataset = dataset[int(0.7*len(dataset)):]
# test_dataset = dataset[900:]

batch_size= 10 # 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)




# # Models and Parameters ##############################################################################################################
device = torch.device('cpu')#('cuda')


dims_dataset = [1,  2, 4, 8, 16, 32, 64, 128, 256, 300]
MC =           [10, 8, 6, 4, 3,  1,  1,  1,   1,   1]

final_metrics = {}

# Early stopping
EPOCHS = 100
PATIENCE = EPOCHS


for dim_dataset in dims_dataset:

    metrics_file = {}

    metrics_file_tot = []

    for time_ in range(MC[dims_dataset.index(dim_dataset)]):

        train_loader = DataLoader(train_dataset.shuffle()[:dim_dataset], batch_size=batch_size, shuffle=True)

        model = Net({'num_enc_steps': 4, 'num_class_steps': 3}).to(device)
        LR = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        metrics_file_temp = {}
        latest_losses = np.array([10000.]*PATIENCE)


        for epoch in tqdm(range(EPOCHS)):
        
            start = time.time()
            metrics = train(device, model, optimizer, train_loader)
            end = time.time()

            # if epoch % 50 == 0:
            val_metrics = evaluate(device, model, val_loader)
            metrics_file_temp[epoch] = {**metrics, **val_metrics}

            latest_losses = shift(latest_losses, -1, cval=np.NaN) 
            latest_losses[PATIENCE - 1] = metrics_file_temp[epoch]['loss/val'] # Loss

            if np.argmin(latest_losses) == 0:
                print(epoch)
                break

        metrics_file_tot.append(metrics_file_temp)

    # Take mean 
    metrics_file = copy(metrics_file_tot[0])
    if len(metrics_file_tot) > 1:

        for epoch in range(EPOCHS):

            for key in metrics_file_tot[0][epoch].keys():

                list_ = [metrics_file_tot[time__][epoch][key] for time__ in range(time_)]
                metrics_file[epoch][key] = sum(list_)/len(list_)

    
    final_metrics[dim_dataset] = metrics_file

# for dim_dataset in dims_dataset:
    
#     train_loader = DataLoader(train_dataset[:dim_dataset], batch_size=batch_size, shuffle=True)

#     model = Net({'num_enc_steps': 4, 'num_class_steps': 3}).to(device)
#     LR = 0.001
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     metrics_file = {}
#     latest_losses = np.array([10000.]*PATIENCE)


#     for epoch in tqdm(range(EPOCHS)):
    
#         start = time.time()
#         metrics = train(device, model, optimizer, train_loader)
#         end = time.time()

#         # if epoch % 50 == 0:
#         val_metrics = evaluate(device, model, val_loader)
#         metrics_file[epoch] = {**metrics, **val_metrics}

#         latest_losses = shift(latest_losses, -1, cval=np.NaN) 
#         latest_losses[PATIENCE - 1] = metrics_file[epoch]['loss/val'] # Loss

#         if np.argmin(latest_losses) == 0:
#             print(epoch)
#             break
    
#     final_metrics[dim_dataset] = metrics_file

    with open(DIR_EXPERIMENT + '/dataset_dimension.txt', 'w') as file:
        file.write(json.dumps(final_metrics))
    np.save(DIR_EXPERIMENT + '/dataset_dimension.npy', final_metrics, allow_pickle = True)



fig = plt.figure(figsize=(25,10))
ax_acc = fig.add_subplot(1, 2, 1)
ax_loss = fig.add_subplot(1, 2, 2)

x_acc = []
y_acc = []

x_loss = []
y_loss = []

plot_loss_acc_epochs_hyperparam(epoch, final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, legend = {'name':'dataset dimension', 'values':dims_dataset}, DIR_EXPERIMENT = DIR_EXPERIMENT)






























































































































































































































































































































