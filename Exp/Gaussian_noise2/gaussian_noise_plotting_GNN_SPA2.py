import numpy as np
import os
from os import walk
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
import scipy.io
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

def nested_dict():
    return defaultdict(nested_dict)

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
dir_dataset = cwd + '/../../../../../Dataset/different_gaussian_noises2/'
datasets_noise = next(walk(dir_dataset), (None, None, []))[2]
datasets_noise = [s for s in datasets_noise if ('dataset' not in s) and ('processed' not in s) and ('raw' not in s) and ('DS_Store' not in s)]
datasets_noise.sort()

final_metrics = {}
values = [0.0001, 0.1, 0.5, 1, 2, 4, 8]
values = [0.0001, 0.1, 0.5, 1, 2]
values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

final_metrics = np.load(DIR_EXPERIMENT + '/gaussian_noises_validation_combinations01_19.npy',  allow_pickle = True).tolist()
y_acc = []


# Load SPA results
SPA_final_metrics_temp = scipy.io.loadmat(DIR_EXPERIMENT + '/accuracy_SPA01_19.mat')['tot_res'][0]
SPA_final_metrics = nested_dict()
SPA_final_metrics_list = []
for i in SPA_final_metrics_temp:
    ris = [el[0] for el in i[0]]
    # print(ris)
    for j in ris:
        SPA_final_metrics[j[0][2][0][0]][j[0][1][0][0]] = j[0][0][0][0]
        if j[0][2][0][0] == j[0][1][0][0]:
            SPA_final_metrics_list.append(j[0][0][0][0])

# Load GNN results
i = 0
for train_data, value in final_metrics.items():
    j = 0
    # print(train_data)
    for valid_data, metrics in value.items():
        # print(valid_data)
        if i == j:
            y_acc.append(metrics['accuracy/val'])
        j += 1
    i += 1
        

fig = plt.figure(figsize=(7.5,7))
ax_acc = fig.add_subplot(1, 1, 1)
# x_acc = values[1:]
x_acc = values[:]

# Plotting
legend = {'name':'$\sigma$ Gaussian noise GNN SPA', 
'values':values, 
'metrics': ['accuracy']}

type_ = legend['name']
VALUES = legend['values']
METRICS = legend['metrics']



acc = [el for el in y_acc]
y_acc = []
for i in range(0, len(values)):
# for i in range(1, len(values)):
    # y_acc.append([acc[i], precision[i], recall[i]])
    y_acc.append([acc[i], SPA_final_metrics_list[i]])


# Draw accuracy
ax_acc.clear()
ax_acc.plot(x_acc, y_acc)

# Format plot
plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
plt.xticks(ticks=x_acc, rotation=45, ha='right', fontsize=14)
plt.subplots_adjust(bottom=0.30, right = 1)
plt.ylim(0.95,1)
ax_acc.set(xlabel=type_)
# ax_acc.title.set_text('Metrics')
ax_acc.grid()
ax_acc.legend(['accuracy GNN', 'accuracy SPA'], loc='lower left')#right', shadow=False)


plt.savefig(DIR_EXPERIMENT + f'/$\sigma$ Gaussian noise GNN SPA 01_19.pdf')
plt.savefig(DIR_EXPERIMENT + f'/$\sigma$ Gaussian noise GNN SPA 01_19.eps', format='eps')
plt.show()





























































































































































































































































































































