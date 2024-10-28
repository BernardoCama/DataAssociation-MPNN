import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 22})

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
from Classes.plotting import plot_loss_acc_values_hyperparam



# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + '/../../../../../Dataset/'
name_dataset = 'bounding_boxes_noise.mat'

final_metrics = {} 
values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

final_metrics = np.load(DIR_EXPERIMENT + '/number_vehicles.npy',  allow_pickle = True).tolist()


# fig = plt.figure(figsize=(12.9,8.32))
# ax_acc = fig.add_subplot(1, 2, 1)
# ax_loss = fig.add_subplot(1, 2, 2)

fig = plt.figure(figsize=(7.5,8.32))
ax_acc = fig.add_subplot(1, 1, 1)
ax_loss = None

x_acc = []
y_acc = []

x_loss = []
y_loss = []


plot_loss_acc_values_hyperparam(final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, 
legend = {'name':'number vehicles', 'values':values, 'metrics': ['accuracy', 'precision', 'recall']}, 
DIR_EXPERIMENT = DIR_EXPERIMENT, 
type_filter = 1,
draw_loss = 0)






























































































































































































































































































































