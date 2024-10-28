import numpy as np
import os
from os import walk
import sys
from importlib import reload
import json
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.loader import DataLoader
import seaborn as sns
import scipy.io
from collections import defaultdict
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
from Classes.dataset import GraphDataset, random_flip_xy
from Classes.model import Net
from Classes.train_validate_fun import *

def nested_dict():
    return defaultdict(nested_dict)


# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + '/../../../../../Dataset/different_noises_rotated/'
datasets_noise = next(walk(dir_dataset), (None, None, []))[2]
datasets_noise = [s for s in datasets_noise if ('dataset' not in s) and ('processed' not in s) and ('raw' not in s) and ('DS_Store' not in s)]
datasets_noise.sort()

final_metrics = nested_dict()
values = ["Pointpillars", "Iso_gaussian","Non_Iso_gaussian","Laplacian", "Uniform","Discrete"]

for name_dataset_noise_train in []:#datasets_noise:

    value_train = name_dataset_noise_train.split('bounding_boxes_noise_')[1].split('.mat')[0].split('_rot')[0]

    # # Models and Parameters ##############################################################################################################
    device = torch.device('cpu')#('cuda')
    del sys.modules['Classes.model'] 
    reload(Classes)
    from Classes.model import Net
    model = Net({'num_enc_steps': 4, 'num_class_steps': 3}).to(device)
    LR = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    metrics_file = {}
    
    EPOCHS = 25
    model.load_state_dict(torch.load(DIR_EXPERIMENT + f'/model_{EPOCHS}_noise_{value_train}'))
    model.eval()

    for name_dataset_noise_val in datasets_noise:

        value_val= name_dataset_noise_val.split('bounding_boxes_noise_')[1].split('.mat')[0].split('_rot')[0]

        raw_data = scipy.io.loadmat(dir_dataset + name_dataset_noise_val)
        number_vehicles = 20
        number_instants = 1498
        raw_data = raw_data['new_dataset'][:number_vehicles,:]
        raw_data = random_flip_xy(raw_data)
        number_instants *= 2

        # import vehicle positions
        # name_dataset_actors = 'actor_data.mat'
        # raw_data_vehicles = scipy.io.loadmat(dir_dataset + '../' + name_dataset_actors)
        # raw_data_vehicles = raw_data_vehicles['vehicles_pos'][0,0]
        # vehicles_pos = np.array([vehicles for vehicles in raw_data_vehicles])
        # # vehicles_pos[vehicle, instant]
        # vehicles_pos_xy = vehicles_pos[:,:,:2]
        # vehicles_pos_xy = {i:vehicles_pos_xy[i] for i in range(vehicles_pos_xy.shape[0])}

        connectivity_matrix_dict = []
        for instant in range(number_instants):
            connectivity_matrix_dict.append([])
            i = 0
            for vehicle in range(number_vehicles):
                connectivity_matrix_dict[instant].append({})
                for name in raw_data[vehicle][instant][0][0][1].reshape([1,-1])[0]:
                    connectivity_matrix_dict[instant][vehicle][name[0]] = i
                    i += 1

        dataset = GraphDataset(root=dir_dataset, raw_data=raw_data, dir_dataset=dir_dataset, number_instants=number_instants, number_vehicles=number_vehicles, connectivity_matrix_dict=connectivity_matrix_dict, name_processed_dataset=name_dataset_noise_val)
        dataset = dataset.shuffle()
        train_dataset = dataset[:int(0.7*len(dataset))]
        # val_dataset = dataset[int(0.7*len(dataset)):]
        val_dataset = dataset[:1600]
        # test_dataset = dataset[900:]

        batch_size= 10 # 1024
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size)


        # Validation
        val_metrics = evaluate(device, model, val_loader)


        print('Epoch: {:03d}, Train: {}, Val: {}, \nValidation: {}\n'.format(EPOCHS, value_train, value_val, val_metrics))
        
        metrics_file = {**val_metrics}


        final_metrics[value_train][value_val] = metrics_file

        with open(DIR_EXPERIMENT + '/different_noises_validation_combinations_rotated.txt', 'w') as file:
            file.write(json.dumps(final_metrics))
        np.save(DIR_EXPERIMENT + '/different_noises_validation_combinations_rotated.npy', final_metrics, allow_pickle = True)


final_metrics = np.load(DIR_EXPERIMENT + '/different_noises_validation_combinations_rotated.npy',  allow_pickle = True).tolist()

matrix = np.random.rand(len(values), len(values))

i = 0
for train_data, value in final_metrics.items():
    j = 0
    for valid_data, metrics in value.items():
        matrix[i][j] = metrics['accuracy/val']
        j += 1
    i += 1

plt.rcParams.update({'font.size': 11}) #  11, 13, 16

fig = plt.figure(figsize=(10,8))

# According to alphabet order
values = ["Discrete", "Iso Gauss","Laplacian", "Non Iso Gauss", "Pointpillars", "Uniform"]

x_axis_labels=values
y_axis_labels=values
sns.heatmap(matrix, annot=True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, fmt='.4g');
plt.xticks(rotation=0) 
plt.yticks(rotation=90, va="center") 
plt.xlabel('Validation dataset');
plt.ylabel('Training dataset');

plt.savefig(DIR_EXPERIMENT + '/matrix_noises_rotated.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/matrix_noises_rotated.eps', format='eps', bbox_inches='tight')

plt.show()


























































































































































































































































