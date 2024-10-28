import numpy as np
import os
from os import walk
import sys
from importlib import reload
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.loader import DataLoader
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
import Classes.model
from Classes.dataset import GraphDataset
from Classes.model import Net
from Classes.train_validate_fun import *
from Classes.plotting import plot_loss_acc_epochs_hyperparam



# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + '/../../../../../Dataset/different_noises/'
datasets_noise = next(walk(dir_dataset), (None, None, []))[2]
datasets_noise = [s for s in datasets_noise if ('dataset' not in s) and ('processed' not in s) and ('raw' not in s) and ('DS_Store' not in s)]
datasets_noise.sort()

final_metrics = {}
values = ["Iso_gaussian","Non_Iso_gaussian","Laplacian", "Uniform","Discrete"]

for name_dataset_noise in datasets_noise:

    value = name_dataset_noise.split('bounding_boxes_noise_')[1].split('.mat')[0]

    raw_data = scipy.io.loadmat(dir_dataset + name_dataset_noise)
    number_vehicles = 20
    number_instants = 1000
    raw_data = raw_data['new_dataset'][:number_vehicles,:]

    # import vehicle positions
    name_dataset_actors = 'actor_data.mat'
    raw_data_vehicles = scipy.io.loadmat(dir_dataset + '../' + name_dataset_actors)
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


    dataset = GraphDataset(root=dir_dataset, raw_data=raw_data, dir_dataset=dir_dataset, number_instants=number_instants, number_vehicles=number_vehicles, connectivity_matrix_dict=connectivity_matrix_dict, name_processed_dataset=name_dataset_noise)
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
    del sys.modules['Classes.model'] 
    reload(Classes)
    from Classes.model import Net
    model = Net({'num_enc_steps': 4, 'num_class_steps': 3}).to(device)
    LR = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)


    metrics_file = {}

    EPOCHS = 50
    for epoch in tqdm(range(EPOCHS)):
        
        start = time.time()
        metrics = train(device, model, optimizer, train_loader)
        end = time.time()
        # print('time train: {}'.format(end-start))
        
        start = time.time()
        val_metrics = evaluate(device, model, val_loader)
        end = time.time()
        # print('time valid: {}'.format(end-start))
        
        # test_metrics = evaluate(test_loader)

        print('Epoch: {:03d}, \nTraining: {}, \nValidation: {}\n'.format(epoch, metrics, val_metrics))
        
        metrics_file[epoch] = {**metrics, **val_metrics}


    final_metrics[value] = metrics_file
    torch.save(model.state_dict(), DIR_EXPERIMENT + f'/model_{EPOCHS}_noise_{value}')


    with open(DIR_EXPERIMENT + '/different_noises.txt', 'w') as file:
        file.write(json.dumps(final_metrics))
    np.save(DIR_EXPERIMENT + '/different_noises.npy', final_metrics, allow_pickle = True)



fig = plt.figure(figsize=(25,10))
ax_acc = fig.add_subplot(1, 2, 1)
ax_loss = fig.add_subplot(1, 2, 2)

x_acc = []
y_acc = []

x_loss = []
y_loss = []

plot_loss_acc_epochs_hyperparam(epoch, final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, legend = {'name':'dataset dimension', 'values':values}, DIR_EXPERIMENT = DIR_EXPERIMENT)






























































































































































































































































































































