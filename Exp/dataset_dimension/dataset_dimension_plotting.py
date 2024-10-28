import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import scipy.io
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
from Classes.train_validate_fun import *



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
        for name in raw_data[vehicle][instant][0][0][1].reshape([1,-1])[0]:
            connectivity_matrix_dict[instant][vehicle][name[0]] = i
            i += 1


# # Models and Parameters ##############################################################################################################
device = torch.device('cpu')#('cuda')


dims_dataset = [1,  2, 4, 8, 16, 32, 64, 128, 256, 300]
MC =           [10, 8, 6, 4, 3,  1,  1,  1,   1,   1]

final_metrics = {}

# Early stopping
EPOCHS = 100


final_metrics = np.load(DIR_EXPERIMENT + '/dataset_dimension.npy',  allow_pickle = True).tolist()


fig = plt.figure(figsize=(12.9,8.32))
ax_acc = fig.add_subplot(1, 2, 1)
ax_loss = fig.add_subplot(1, 2, 2)

x_acc = []
y_acc = []

x_loss = []
y_loss = []

# plot_loss_acc_epochs_hyperparam(EPOCHS, final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, legend = {'name':'dataset dimension', 'values':dims_dataset}, DIR_EXPERIMENT = DIR_EXPERIMENT, type_filter = 2)

for value in dims_dataset:
    print(f'value: {value}, acc: {final_metrics[value][EPOCHS-1]["accuracy/val"]:.3f}, precision: {final_metrics[value][EPOCHS-1]["precision/val"]:.3f}, recall: {final_metrics[value][EPOCHS-1]["recall/val"]:.3f}')
    





























































































































































































































































































































