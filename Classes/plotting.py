import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
# plt.rcParams.update({'font.size': 22})
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
import random
from copy import copy
from torch_geometric.utils import to_networkx



# Plot training/validation accuracies and losses during training
def plot_loss_acc(epoch, metrics, val_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, DIR_EXPERIMENT, type_filter = None):

    if type_filter == None:
        type_filter = 0
    
    x_loss.append(epoch)  
    x_acc.append(epoch)
    
    tot_loss = {'loss/train': metrics['loss/train'],'loss/val': val_metrics['loss/val']}
    y_loss.append([v for k, v in tot_loss.items()])
    
    t = metrics.popitem()
    t = val_metrics.popitem()
    
    tot_metrics = {**metrics, **val_metrics}
    y_acc.append([v for k, v in tot_metrics.items()])

    
    # Draw accuracy
    ax_acc.clear()
    ax_acc.plot(x_acc, y_acc)

    # Format plot
    plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    # plt.ylim(0.7,1)
    ax_acc.set(xlabel='Epoch')
    # ax_acc.title.set_text('Metrics')
    ax_acc.grid()
    ax_acc.legend([k for k,v in tot_metrics.items()], loc='lower right', shadow=False)
    
    
    # Draw loss
    ax_loss.clear()
    ax_loss.plot(x_loss, y_loss)

    # Format plot
    plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    ax_loss.set(xlabel='Epoch', ylabel='Loss')
    # ax_loss.title.set_text('Loss')
    ax_loss.grid()
    ax_loss.legend([k for k,v in tot_loss.items()], shadow=False)
    
    plt.savefig(DIR_EXPERIMENT + '/training_epochs.pdf', bbox_inches='tight')
    plt.savefig(DIR_EXPERIMENT + '/training_epochs.eps', format='eps', bbox_inches='tight')
    # plt.show()



# Plot training/validation accuracies and losses after training
def plot_loss_acc_after_training(num_epochs, final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, DIR_EXPERIMENT, type_filter = None, print_legend = None, only_valid = None):

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if type_filter == None:
        type_filter = 0
    if print_legend is None:
        print_legend = 0
    if only_valid is None:
        only_valid = 0
    
    for epoch in range(num_epochs):
        x_loss.append(epoch)  
        x_acc.append(epoch)
        
        if only_valid:
            y_loss.append([final_metrics[epoch]['loss/val']])
        else:
            y_loss.append([final_metrics[epoch]['loss/train'], final_metrics[epoch]['loss/val']])

        
        t = final_metrics[epoch].pop('loss/train')
        t = final_metrics[epoch].pop('loss/val')
        
        if only_valid: 
            y_acc.append([v for k, v in final_metrics[epoch].items() if 'val' in k])
        else:
            y_acc.append([v for k, v in final_metrics[epoch].items()])

    # Draw accuracy
    ax_acc.clear()
    ax_acc.plot(x_acc, y_acc)

    # Format plot
    plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    plt.ylim(0.8,1)
    ax_acc.set(xlabel='Epoch')
    # ax_acc.title.set_text('Metrics')
    ax_acc.grid()
    if print_legend:
        ax_acc.legend([k for k in final_metrics[0].keys() if 'val' in k] if only_valid else [k for k in final_metrics[0].keys()], loc='lower right', shadow=False)
    
    
    # Draw loss
    ax_loss.clear()
    ax_loss.plot(x_loss, y_loss)

    # Format plot
    plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    ax_loss.set(xlabel='Epoch', ylabel='Loss')
    # ax_loss.title.set_text('Loss')
    ax_loss.grid()
    if print_legend:
        if only_valid:
            ax_loss.legend(['loss/val'], shadow=False)
        else:
            ax_loss.legend(['loss/train', 'loss/val'], shadow=False)
    
    plt.savefig(DIR_EXPERIMENT + '/training_epochs.pdf', bbox_inches='tight')
    plt.savefig(DIR_EXPERIMENT + '/training_epochs.eps', format='eps', bbox_inches='tight')
    plt.show()


def plot_error_bar(final_metrics, x, log_path, which_epoch = -1, file_name = '', MC_inter = 2, num_epochs = 100, xlim = None, ylim = None, logy = 0, logx = 0, save_eps = 0, ticks = None, ax = None, save_pdf = 1, plt_show = 1, axis_off = 0, plot_vertical_lines = 0):
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    x_acc = []
    y_acc = []
    y_recall = []
    y_precision = []

    VALUES = ticks

    for i, epoch in enumerate(range(num_epochs)):

        x_acc.append(epoch)

        y_acc.append([])
        y_recall.append([])
        y_precision.append([])

        tot_loss = []
        for j, value in enumerate(VALUES):
            y_acc[i].append([])
            y_recall[i].append([])
            y_precision[i].append([])
            for iter in range(MC_inter):
                y_acc[i][j].append(final_metrics[value][iter][epoch]['accuracy/val'])
                y_recall[i][j].append(final_metrics[value][iter][epoch]['recall/val'])
                y_precision[i][j].append(final_metrics[value][iter][epoch]['precision/val'])

    y_acc = np.array(y_acc)
    y_recall = np.array(y_recall)
    y_precision = np.array(y_precision)

    # Take last epoch
    y_acc = np.squeeze(y_acc[which_epoch,:,:])
    y_recall = np.squeeze(y_recall[which_epoch,:,:])
    y_precision = np.squeeze(y_precision[which_epoch,:,:])

    y_acc_mean = np.mean(y_acc,1)
    std_values = [np.std(arr) for arr in y_acc]
    y_acc_mean_plus_std = [std for mean, std in zip(y_acc_mean, std_values)]
    y_acc_mean_minus_std = [std for mean, std in zip(y_acc_mean, std_values)]
    y_acc_err = np.array([y_acc_mean_minus_std, # 'down' error
                y_acc_mean_plus_std])  # 'up' error

    y_recall_mean = np.mean(y_recall,1)
    std_values = [np.std(arr) for arr in y_recall]
    y_recall_mean_plus_std = [std for mean, std in zip(y_recall_mean, std_values)]
    y_recall_mean_minus_std = [std for mean, std in zip(y_recall_mean, std_values)]
    y_recall_err = [y_recall_mean_minus_std, # 'down' error
                    y_recall_mean_plus_std]  # 'up' error
    
    y_precision_mean = np.mean(y_precision,1)
    std_values = [np.std(arr) for arr in y_precision]
    y_precision_mean_plus_std = [std for mean, std in zip(y_precision_mean, std_values)]
    y_precision_mean_minus_std = [std for mean, std in zip(y_precision_mean, std_values)]
    y_precision_err = [y_precision_mean_minus_std, # 'down' error
                        y_precision_mean_plus_std]  # 'up' error
    

    # Desired figure and axes size (in inches)
    fig_width_inches = 6.4
    fig_height_inches = 4.8
    axes_width_inches = 4.96
    axes_height_inches = 3.5

    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches)) # default size

    # Plot accuracy
    plt.errorbar(x, y_acc_mean, yerr=y_acc_err, capsize=3, fmt="--o", color=colors[0], ecolor=colors[0])
    # Plot precision
    plt.errorbar(x, y_precision_mean, yerr=y_precision_err, capsize=3, fmt="--o", color=colors[1], ecolor=colors[1])
    # Plot recall
    plt.errorbar(x, y_recall_mean, yerr=y_recall_err, capsize=3, fmt="--o", color=colors[2], ecolor=colors[2])

    plt.legend(['Accuracy', 'Precision', 'Recall'], loc='best', shadow=False)
    # plt.grid()

    if plot_vertical_lines:
        for _ in x:
            plt.axvline(_, color='black', linestyle = '--', linewidth = 1, alpha = 0.5)

    if logy:
        plt.yscale('log', nonposy='clip')
    if logx:
        plt.xscale('log')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if ticks is not None:
        plt.xticks(ticks)  # Generate ticks from 1 to 10, inclusive, with a step of 1
 

    if axis_off:
        plt.xticks([], [])
        plt.yticks([], [])
    
    # adjust axes size
    # Initialize the loop variables
    current_axes_width = 0
    current_axes_height = 0
    delta = 0.001
    left, right, bottom, top = 0, 1, 0, 1
    # Loop until the desired axes size is met
    while abs(current_axes_width - axes_width_inches) > 0.01 or abs(current_axes_height - axes_height_inches) > 0.01:
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        
        ax = plt.gca()
        ax_position = ax.get_position()
        
        current_axes_width = fig_width_inches * (ax_position.x1 - ax_position.x0)
        current_axes_height = fig_height_inches * (ax_position.y1 - ax_position.y0)
        
        if current_axes_width > axes_width_inches:
            right -= delta
        elif current_axes_width < axes_width_inches:
            right += delta
            
        if current_axes_height > axes_height_inches:
            top -= delta
        elif current_axes_height < axes_height_inches:
            top += delta

    if save_pdf:
        plt.savefig(log_path + f'/{file_name}.pdf', bbox_inches='tight')
    if save_eps:
        plt.savefig(log_path + f'/{file_name}.eps', format='eps', bbox_inches='tight')
    if plt_show:
        plt.show()


# Plot validation loss and accuracy varying the epochs to choose hyperparameters
def plot_loss_acc_epochs_hyperparam(num_epochs, final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, legend = None, DIR_EXPERIMENT = None, type_filter = None, draw_var = None, draw_loss = None):
    
    type_ = legend['name']
    VALUES = legend['values']

    for epoch in range(num_epochs):

        x_loss.append(epoch)
        x_acc.append(epoch)

        # tot_loss = {'loss/train': metrics_file[epoch]['loss/train'],'loss/val': metrics_file[epoch]['loss/val']}
        tot_loss = []
        for value in VALUES:
            tot_loss += [final_metrics[value][epoch]['loss/val']]
        y_loss.append(tot_loss)

        # tot_loss = {'loss/val': [final_metrics[value][epoch]['loss/val'] for value in VALUES]}
        # y_loss.append([v for k, v in tot_loss.items()])
        
        for value in VALUES:
            t = final_metrics[value][epoch].pop('loss/train', None)
            t = final_metrics[value][epoch].pop('loss/val', None)

            t = final_metrics[value][epoch].pop('accuracy/train', None)
            t = final_metrics[value][epoch].pop('precision/train', None)
            t = final_metrics[value][epoch].pop('recall/train', None)

            t = final_metrics[value][epoch].pop('precision/val', None)
            t = final_metrics[value][epoch].pop('recall/val', None)            
        
        tot_metrics = final_metrics
        
        
        temp = []
        for value in VALUES:
            temp += [v for k, v in tot_metrics[value][epoch].items()]

        y_acc.append(temp)


    y_acc_not_filtered = copy(y_acc)
    
    y_acc = np.array(y_acc)
    y_acc_not_filtered = np.array(y_acc_not_filtered)
    yerr = np.abs(y_acc_not_filtered - y_acc)
    
    # Draw accuracy
    ax_acc.clear()
    ax_acc.plot(x_acc, y_acc)

    # Draw variances
    if draw_var == 1:
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for value in range(len(VALUES)):
            ax_acc.fill_between(x_acc, y_acc[:,value]-yerr[:,value], y_acc[:,value]+yerr[:,value],facecolor=cycle[value],alpha=0.5)

    # Format plot
    plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    plt.ylim(0.92,1)
    ax_acc.set(xlabel='Epoch', ylabel='Accuracy')
    # ax_acc.title.set_text('Metrics')
    ax_acc.grid()
    ax_acc.legend(VALUES, loc='lower right', shadow=False)
    
    
    # Draw loss
    if draw_loss == 1:
        ax_loss.clear() 
        ax_loss.plot(x_loss, y_loss)

        # Format plot
        plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30, right = 1)
        ax_loss.set(xlabel='Epoch', ylabel='Loss')
        # ax_loss.title.set_text('Loss')
        ax_loss.grid()
        ax_loss.legend(VALUES, shadow=False)
    
    plt.savefig(DIR_EXPERIMENT + '/training_epochs_hyperparam.pdf', bbox_inches='tight')
    plt.savefig(DIR_EXPERIMENT + '/training_epochs_hyperparam.eps', format='eps', bbox_inches='tight')
    plt.show()


# Plot validation loss and accuracy/recall/precision varying the values of hyperparameters
def plot_loss_acc_values_hyperparam(final_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, legend = None, DIR_EXPERIMENT = None, type_filter = None, draw_loss = None, ylim_ = None, ylog = None):
    
    type_ = legend['name']
    VALUES = legend['values']
    METRICS = legend['metrics']

    for value in VALUES:

        x_loss.append(value)
        x_acc.append(value)

        y_loss.append(final_metrics[value]['loss/val'])

        temp = []
        for metric in METRICS:
            temp += [final_metrics[value][f'{metric}/val']]
        y_acc.append(temp)

    # Draw accuracy
    ax_acc.clear()
    ax_acc.plot(x_acc, y_acc)

    if ylim_ is not None and ylim_:       
        ax_acc.set_ylim(ylim_)
    if ylog is not None and ylog:
        ax_acc.set_yscale('log')

    # Format plot
    plt.sca(ax_acc)   # Use the pyplot interface to change just one subplot
    plt.xticks(ticks=x_acc, rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    # plt.ylim(0.7,1)
    ax_acc.set(xlabel=type_)
    # ax_acc.title.set_text('Metrics')
    ax_acc.grid()
    legend = ax_acc.legend(METRICS, loc='best')#right', shadow=False)
    # SET POSITON OF LEGEND
    # legend = ax_acc.legend(METRICS, bbox_to_anchor=(0.008380952380952406, 0.8813716170009948, 0.13596190476190476, 0.10723083657990706),
    #                     loc='upper left', bbox_transform=ax_acc.transAxes)

    # Draw loss
    if draw_loss == 1:
        ax_loss.clear()
        ax_loss.plot(x_loss, y_loss)

        # Format plot
        plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
        plt.xticks(ticks=x_loss, rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30, right = 1)
        ax_loss.set(xlabel=type_, ylabel='Loss')
        # ax_loss.title.set_text('Loss')
        ax_loss.grid()
        # ax_loss.legend('loss/val')
    
    plt.savefig(DIR_EXPERIMENT + f'/{type_}.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(DIR_EXPERIMENT + f'/{type_}.eps', format='eps', bbox_inches='tight', pad_inches=0)
    plt.show()


    # GET POSITION OF LEGEND
    # Get the legend position in display coordinates
    # legend_bbox = legend.get_window_extent()
    # # Transform the display coordinates to axes coordinates
    # legend_bbox_axes = ax_acc.transAxes.inverted().transform(legend_bbox)

    # # Calculate the width and height of the bounding box
    # width = legend_bbox_axes[1, 0] - legend_bbox_axes[0, 0]
    # height = legend_bbox_axes[1, 1] - legend_bbox_axes[0, 1]

    # # Create a legend for ax2 at the same position as ax1's legend
    # legend = ax_acc.legend(METRICS, bbox_to_anchor=(legend_bbox_axes[1, 1], legend_bbox_axes[1, 0], width, height),
    #                     loc='upper right', bbox_transform=ax_acc.transAxes)


# Plot loss as function of dataset dimension
def plot_dataset_loss(fig, key, value, x_loss, y_loss, ax_loss, DIR_EXPERIMENT):
    
    #for key, value in final_metrics.items():

    x_loss.append(key)
    tot_loss = {'loss/train': value['loss/train'],'loss/val': value['loss/val']}
    y_loss.append([v for k, v in tot_loss.items()])
    #y_loss_ = y_loss

    window_size = 4
    train_ = [el[0] for el in y_loss]
    numbers_series = pd.Series(train_)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    train_ = moving_averages_list[window_size - 1:]

    val_ = [el[1] for el in y_loss]
    numbers_series = pd.Series(val_)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    val_ = moving_averages_list[window_size - 1:]

    y_loss_ = list(zip(train_, val_))

    # Draw loss
    ax_loss.clear()
    ax_loss.plot(x_loss[:len(y_loss_)], y_loss_)

    # Format plot
    plt.sca(ax_loss)  # Use the pyplot interface to change just one subplot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30, right = 1)
    ax_loss.set(xlabel='Size training dataset', ylabel='Loss')
    # ax_loss.title.set_text('Losses')
    ax_loss.grid()
    ax_loss.legend([k for k,v in tot_loss.items()], shadow=False)   
    
    plt.savefig(DIR_EXPERIMENT + '/dataset_loss.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(DIR_EXPERIMENT + '/dataset_loss.eps', format='eps', bbox_inches='tight')
#     fig.canvas.draw_idle()
#     plt.draw()
#     fig.clear()
#     plt.show()



# Draw graph of the sample input with edge classification specified by labels
def draw_graph (sample, labels, colors = None, type_= None, DIR_EXPERIMENT = None):
    plt.rcParams.update({'font.size': 10})

    options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}

    # Create Graph
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False)

    # Layout
    pos=nx.circular_layout(G)

    # Recover time instant
    instant = sample.time_instant
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(np.unique(node_vehicle)))]
    # Draw nodes
    for node in range(number_nodes):
        which_vehicle = node_vehicle[node]
        nx.draw_circular(G, edgelist = [], nodelist=[node], node_color=np.array([colors[int(which_vehicle)]]), **options)
        # nx.draw_networkx_nodes(G, pos,  nodelist=[node], node_color=np.array([colors[int(which_vehicle)]]), **options)


    # Draw edges
    nx.draw_circular(G,  nodelist=[], **options)
    edges = sample.edge_index.t()
    # labels = sample.edge_labels.detach().numpy()
    edgelist_ones = edges[np.where(labels==1)[0]].numpy()
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy()

    nx.draw_circular(
        G,
        edgelist=edgelist_ones,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:blue",
    )
    nx.draw_circular(
        G,
        edgelist=edgelist_zeros,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:red",
    )


    # Legend
    # legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
    #                    Line2D([0], [0], marker='o', color='w', label='Scatter',
    #                           markerfacecolor='g', markersize=15),
    #                    Patch(facecolor='orange', edgecolor='r',
    #                          label='Color Patch')]
    legend_elements = [Line2D([0], [0], color="tab:blue", lw=4, label='Association'),
                       Line2D([0], [0], color="tab:red", lw=4, label='No Association')]
    legend1 = plt.legend(handles=legend_elements, loc='upper right', shadow=False)

    elements = []
    for vehicle in range(len(colors)):
        elements.append(Line2D([0], [0], color=colors[int(vehicle)], lw=4, label=f'Vehicle{vehicle}'))

    legend2 = plt.legend(handles=elements, loc='upper left', shadow=False)
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    if type_ == 'GT':
        # plt.title("Classification edges GT")
        plt.savefig(DIR_EXPERIMENT + '/graph_gt.pdf', bbox_inches='tight')
        plt.savefig(DIR_EXPERIMENT + '/graph_gt.eps', format = 'eps', bbox_inches='tight')
    else:
        # plt.title("Classification edges Prediction")
        plt.savefig(DIR_EXPERIMENT + '/graph_prediction.pdf', bbox_inches='tight')
        plt.savefig(DIR_EXPERIMENT + '/graph_prediction.eps', format = 'eps', bbox_inches='tight')
        
    plt.rcParams.update({'font.size': 22})
    
    return colors


def draw_graph_some_edges (sample, labels, colors = None, type_= None, which_edges = None, DIR_EXPERIMENT=None):
    # Draw only edges specified by which_edges 
    
    plt.rcParams.update({'font.size': 10})
    
    # Draw only some edges
    edges = sample.edge_index.t()
    if which_edges is not None:
        edges = edges[which_edges]
        labels = labels[which_edges]

    options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}

    # Create Graph
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False)

    # Layout
    pos=nx.circular_layout(G)

    # Recover time instant
    instant = sample.time_instant
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    # node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(np.unique(node_vehicle)))]
    # Draw nodes
    for node in range(number_nodes):
        which_vehicle = node_vehicle[node]
        if node in edges.flatten():
            nx.draw_circular(G, edgelist = [], nodelist=[node], node_color=np.array([colors[int(which_vehicle)]]), **options)
            # nx.draw_networkx_nodes(G, pos,  nodelist=[node], node_color=np.array([colors[int(which_vehicle)]]), **options)

    # Draw edges
    nx.draw_circular(G, edgelist = [], nodelist=[], **options)
    # labels = sample.edge_labels.detach().numpy()
    edgelist_ones = edges[np.where(labels==1)[0]].numpy()
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy()

    nx.draw_circular(
        G,
        edgelist=edgelist_ones,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:blue",
    )
    nx.draw_circular(
        G,
        edgelist=edgelist_zeros,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:red",
    )


    # Legend
    # legend_elements = [Line2D([0], [0], color='b', lw=4, label='Line'),
    #                    Line2D([0], [0], marker='o', color='w', label='Scatter',
    #                           markerfacecolor='g', markersize=15),
    #                    Patch(facecolor='orange', edgecolor='r',
    #                          label='Color Patch')]
    legend_elements = [Line2D([0], [0], color="tab:blue", lw=4, label='Association'),
                       Line2D([0], [0], color="tab:red", lw=4, label='No Association')]
    legend1 = plt.legend(handles=legend_elements, loc='upper right')

    elements = []
#     vehicles = [node_vehicle[node] for node in edges.flatten()]
#     for vehicle in range(len(colors)):
#         if vehicle in vehicles:
#             elements.append(Line2D([0], [0], color=colors[int(vehicle)], lw=4, label=f'Vehicle{vehicle}'))

    legend2 = plt.legend(handles=elements, loc='upper left')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    if type_ == 'GT':
        plt.title("Misclassified edges GT")
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_gt.png', bbox_inches='tight')
    else:
        plt.title("Misclassified edges Prediction")
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_prediction.png', bbox_inches='tight')
    
    plt.rcParams.update({'font.size': 22})
    return colors






# Draw the connected components of the graph and with edge classification specified by labels
def draw_graph_connected_components_wrong_edges (sample, labels, colors = None, type_= None, which_edges = None, DIR_EXPERIMENT=None):
    # Highlight edges specified by which_edges 
    
    # Draw only some edges
    edges = sample.edge_index.t()
    if which_edges is not None:
        edges = edges[which_edges]
        labels = labels[which_edges]
    
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(np.unique(node_vehicle)))]

    which_vehicles = [node_vehicle[node] for node in range(number_nodes)]
    color = np.array([colors[int(which_vehicle)] for which_vehicle in which_vehicles])
    
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False)
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    plt.figure(figsize=(15, 15))
    nx.draw(G, pos, node_size=100, alpha=0.5, node_color=color, with_labels=False)
    
    edgelist_ones = edges[np.where(labels==1)[0]].numpy()
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy()

    nx.draw(
        G,
        pos,
        edgelist=edgelist_ones,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:blue",
    )
    nx.draw(
        G,
        pos,
        edgelist=edgelist_zeros,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:red",
    )
    
    plt.axis("equal")
    
    if type_ == 'GT':
        # plt.title("Misclassified edges GT")
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_connected_components_gt.pdf', bbox_inches='tight')
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_connected_components_gt.eps', format='eps', bbox_inches='tight')
    else:
        # plt.title("Misclassified edges Prediction")
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_connected_components_prediction.pdf', bbox_inches='tight')
        plt.savefig(DIR_EXPERIMENT + '/graph_misclassified_edges_connected_components_prediction.eps', format='eps', bbox_inches='tight')
    
    return colors    



def draw_graph_connected_components (sample, labels, colors = None, type_= None, which_edges = None, DIR_EXPERIMENT=None):
    # Highlight edges specified by which_edges 
    
    # Draw only some edges
    edges = sample.edge_index.t()
    if which_edges is not None:
        edges = edges[which_edges]
        labels = labels[which_edges]
    
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    # node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(np.unique(node_vehicle)))]

    vehicles = np.unique(node_vehicle).astype(int)
    color_nodes = np.array([colors[int(node_vehicle[int(node)])] for node in range(number_nodes)])
    
    edgelist_ones = edges[np.where(labels==1)[0]].numpy()
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy()
    
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
    
    # Consider only edges classified as 1
    G.remove_edges_from([tuple(x) for x in edgelist_zeros])
    
    pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
    plt.figure(figsize=(15, 15))
    nx.draw(G, pos, edgelist=[], node_size=100, alpha=1, node_color=color_nodes, with_labels=False)
    

    nx.draw(
        G,
        pos,
        edgelist=edgelist_ones,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:blue",
    )
#     nx.draw(
#         G,
#         pos,
#         edgelist=edgelist_zeros,
#         nodelist=[],
#         width=3,
#         alpha=0.5,
#         edge_color="tab:red",
#     )
    
    plt.axis("equal")

    
    legend_elements = [Line2D([0], [0], color="tab:blue", lw=4, label='Association'),
                       Line2D([0], [0], color="tab:red", lw=4, label='No Association')]
    legend1 = plt.legend(handles=legend_elements, loc='upper right')

    elements = []
    for vehicle in vehicles:
        elements.append(Line2D([0], [0], color=colors[int(vehicle)], lw=4, marker='o', markersize = 10, linestyle="None", label=f'Vehicle{vehicle}'))

    legend2 = plt.legend(handles=elements, loc='upper left')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    if type_ == 'GT':
        plt.title("Classification edges connected components GT")
        plt.savefig(DIR_EXPERIMENT + '/graph_gt_connected_components.png', bbox_inches='tight')
    else:
        plt.title("Classification edges connected components Prediction")
        plt.savefig(DIR_EXPERIMENT + '/graph_prediction_connected_components.png', bbox_inches='tight')
        
    plt.rcParams.update({'font.size': 22})  
    
    return colors


def draw_vehicle_features (sample, labels, colors = None, type_= None, which_edges = None, gt = None, image_path = None, plot_fake_vehicles = None, vehicles_pos_xy=None, pedestrians_pos_xy=None, DIR_EXPERIMENT=None, save_eps = None, save_pdf = None, save_jpg = None, title_savename = None):
    # sample:             sample of the graph
    # lables:             prediction of association
    # colors:             colors for the vehicles 
    # type_:              'GT' if labels are of the gt, 'Prediction' otherwise
    # gt:                 gt of prediction
    # image_path:         path to background image
    # plot_fake_vehicles: if we have to plot the fake vehicles for association in time 
    
    
    if save_eps is None:
        save_eps = 0
    
    if save_pdf is None:
        save_pdf = 0

    if save_jpg is None:
        save_jpg = 0

    # Take instants of fake vehicles
    if plot_fake_vehicles is not None:
        fake_vehicles_instants = list(plot_fake_vehicles.keys())
        
    #{sample.time_instant.numpy()[0]:{'x_new':x_new, 'track_id_new':track_id_new}}
    plt.rcParams.update({'font.size': 20})
    
    
    # Draw only some edges
    edges = sample.edge_index.t()
    if which_edges is not None:
        edges = edges[which_edges]
        labels = labels[which_edges]  
        if gt is not None:
            edges_gt = edges[which_edges]
            labels_gt = labels[which_edges]            
    
    # Time instant
    instant = int(sample.time_instant)
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # print(node_vehicle)
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Node attributes
    x = sample.x.reshape(-1, 3, 8).numpy()
    centroids = np.mean(x, axis=2)
    # Recover feature_name associated to the node
    feature_name = sample.feature_name
        

    edgelist_ones = edges[np.where(labels==1)[0]].numpy() # does not preserve order
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy() 
    
    if gt is not None:
        edgelist_ones_gt = edges[np.where(gt==1)[0]].numpy() # does not preserve order
        edgelist_zeros_gt = edges[np.where(gt==0)[0]].numpy() 
      
        
    # Create graph
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
    # Consider only edges classified as 1
    G.remove_edges_from([tuple(x) for x in edgelist_zeros]) 
    # Compute connected components
    G_undirected = G.to_undirected(reciprocal=True)
    connected_components = nx.connected_components(G_undirected)
    connected_components = sorted(connected_components)
    print(f'Number connected components: {len(connected_components)}')
    G_components = [G_undirected.subgraph(c).copy() for c in connected_components]
    
    if gt is not None:
        # Create graph
        G_gt = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
        # Consider only edges classified as 1
        G_gt.remove_edges_from([tuple(x) for x in edgelist_zeros_gt]) 
        # Compute connected components
        G_undirected_gt = G_gt.to_undirected(reciprocal=True)
        connected_components_gt = nx.connected_components(G_undirected_gt)
        connected_components_gt = sorted(connected_components_gt)
        G_components_gt = [G_undirected_gt.subgraph(c).copy() for c in connected_components_gt]
        
    
    # Recover real id of vehicles
    node_vehicle = encoder.inverse_transform(node_vehicle).astype(int)
    # print(node_vehicle)
    
    # Recover correspondence node-vehicle and compute graph vehicle-features
    G_vehicle_features = nx.Graph()
    connected_components_vehicles = []
    # Which vehicles are in each connected component
    connected_components_unique_vehicles = []
    # new name of each target
    name_features = []
    # mean position of target
    pos_features = {}
    i = 0
    for el in connected_components:
        el = sorted(el)
        # print(el)
        print(el, np.array([feature_name[int(node)] for node in el]), np.array([node_vehicle[int(node)] for node in el]))
        connected_components_vehicles.append(np.array([node_vehicle[int(node)] for node in el])) 
        connected_components_unique_vehicles.append(np.unique(np.array([node_vehicle[int(node)] for node in el])))
        if type_ == 'GT' and gt is None:
            name_features.append(f'{feature_name[el[0]]}')
        else:
            name_features.append(f'ped_pred_{i}')
        pos_features[name_features[i]] = np.mean([centroids[int(node)] for node in el], 0)
        # print(pos_features[name_features[i]])
        # print('\n')
  
        G_vehicle_features.add_node(name_features[i])
        edges_to_add = [ tuple([name_features[i], vehicle])  for vehicle in connected_components_unique_vehicles[i]]
        G_vehicle_features.add_edges_from(edges_to_add)
        
        i+=1
        
    if gt is not None:
        j = 0
        connected_components_vehicles_gt = []
        # Which vehicles are in each connected component
        connected_components_unique_vehicles_gt = []
        for el in connected_components_gt:
            el = sorted(el)
            # print(el)
            # print(el, np.array([feature_name[node] for node in el]),  np.array([node_vehicle[node] for node in el]))
            connected_components_vehicles_gt.append(np.array([node_vehicle[int(node)] for node in el])) 
            connected_components_unique_vehicles_gt.append(np.unique(np.array([node_vehicle[int(node)] for node in el])))
            name_features.append(f'ped_gt_{j}')
            pos_features[name_features[i]] = np.mean([centroids[int(node)] for node in el], 0)

            G_vehicle_features.add_node(name_features[i])
            # edges_to_add = [ tuple([name_features[i], vehicle])  for vehicle in connected_components_unique_vehicles[i]]
            # G_vehicle_features.add_edges_from(edges_to_add)

            i+=1
            j+=1
        
    
    # Find all vehicles names
    vehicles = list(G_vehicle_features.nodes)
    for element in name_features:
        if element in vehicles:
            vehicles.remove(element)        
    # print (vehicles)
            
    # vehicles_pos_xy
    pos = {}
    for v in vehicles:
        # print(v, instant)
        pos[v] = tuple([-vehicles_pos_xy[v][instant][1], vehicles_pos_xy[v][instant][0]]) # ACCORDING TO REF SYS
        # print(v, pos[v])
    for f in name_features:
        # mean of centroids 
        pos[f] = tuple([-pos_features[f][1], pos_features[f][0] ])  # ACCORDING TO REF SYS

    if pedestrians_pos_xy is not None:
        for k in pedestrians_pos_xy.keys():
            # Add real position of pedestrians
            G_vehicle_features.add_node(k)
            pos[k] = tuple([-pedestrians_pos_xy[k][instant][1], pedestrians_pos_xy[k][instant][0]]) # ACCORDING TO REF SYS
        
        
    
    # pos = nx.nx_agraph.graphviz_layout(G_vehicle_features, prog="twopi", args="")
    fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(111)
    
    # Check corrected association of one target by all vehicles
    if gt is not None:
        correct_features = []
        for p in [k for k in name_features if 'pred' in k]:
            for g in [k for k in name_features if 'gt' in k]:
                # if the feature is correct assigned to vehicles 
                if pos[p]==pos[g]:
                    correct_features.append(g)
                    correct_features.append(p)
                    
        # Find wrong associations
        wrong_features = list(G_vehicle_features.nodes)
        for element in pos.keys():
            if (element in vehicles) or (element in correct_features):
                wrong_features.remove(element) 
                
    
    # Insert in graph the fake vehicle and the targets
    fake_edges = []
    if plot_fake_vehicles is not None:
        
        for instant in fake_vehicles_instants:
            
            G_vehicle_features.add_node(1000 + int(instant))
  
            edges_to_add = [ tuple([1000 + int(instant), target]) for target in plot_fake_vehicles[instant]['track_id_new']]
            fake_edges += edges_to_add
            G_vehicle_features.add_edges_from(edges_to_add)
        
            for i in range(len(plot_fake_vehicles[instant]['track_id_new'])):
                
                centroid = np.mean(plot_fake_vehicles[instant]['x_new'][i], axis=1)
                pos[plot_fake_vehicles[instant]['track_id_new'][i]] = tuple([-centroid[1], centroid[0]])
                
            pos[1000 + int(instant)] = tuple([-210, 100+10*fake_vehicles_instants.index(instant)])
            
            
    # Compute random colors
    if colors == None:
        colors = [[random.random(), random.random(), random.random()] for vehicle in range(len(np.unique(node_vehicle)))]
    color = np.array([colors[int(vehicle)] for vehicle in vehicles])
        
    
    
    # plot nodes
    if gt is None:
        # features predicted 
        nx.draw(G_vehicle_features, pos, nodelist=[k for k in name_features if 'FP' in k], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='black', with_labels=False)  
        nx.draw(G_vehicle_features, pos, nodelist=[k for k in name_features if 'FP' not in k], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='blue', with_labels=False)   
        # vehicles
        nx.draw(G_vehicle_features, pos, nodelist=vehicles, edgelist=[], node_shape="s", node_size=100, alpha=1, node_color=color, with_labels=False)
    else: 
        
        # print(wrong_features, correct_features)
        # correct features
        nx.draw(G_vehicle_features, pos, nodelist=correct_features, edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='green', with_labels=False)         
        # wrong features gt
        nx.draw(G_vehicle_features, pos, nodelist=[k for k in wrong_features if 'gt' in k], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='red', with_labels=False) 
        # wrong features predicted
        nx.draw(G_vehicle_features, pos, nodelist=[k for k in wrong_features if 'pred' in k], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='blue', with_labels=False) 
        # features gt 
        # nx.draw(G_vehicle_features, pos, nodelist=[k for k in name_features if 'gt' in k], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='red', with_labels=False) 
        # vehicles
        nx.draw(G_vehicle_features, pos, nodelist=vehicles, edgelist=[], node_shape="s", node_size=100, alpha=1, node_color=color, with_labels=False)  
    
    if pedestrians_pos_xy is not None:
        nx.draw(G_vehicle_features, pos, nodelist=list(pedestrians_pos_xy.keys()), edgelist=[], node_shape="x", node_size=100, alpha=0.5, node_color='black', with_labels=False) 
        
    if plot_fake_vehicles is not None:

        # vehicles
        i = 1
        for instant in fake_vehicles_instants:
            nx.draw(G_vehicle_features, pos, nodelist=[1000 + int(instant)], edgelist=[], node_shape="s", node_size=200, alpha=i/len(fake_vehicles_instants), node_color='orange', with_labels=False) 
            
            # features previous times
            nx.draw(G_vehicle_features, pos, nodelist=[target for target in plot_fake_vehicles[instant]['track_id_new']], edgelist=[], node_shape="^", node_size=100, alpha=0.5, node_color='orange', with_labels=False) 
            
            i += 1
        

    # plot edges
    edges_to_draw = list(G_vehicle_features.edges())
    
    if plot_fake_vehicles is not None:
        for element in fake_edges:
            if element in edges_to_draw:
                edges_to_draw.remove(element)           
        nx.draw(
            G_vehicle_features,
            pos,
            edgelist = fake_edges,
            nodelist=[],
            width=3,
            alpha=0.5,
            edge_color="tab:orange",
        )
          
    nx.draw(
        G_vehicle_features,
        pos,
        edgelist = edges_to_draw,
        nodelist=[],
        width=3,
        alpha=0.5,
        edge_color="tab:gray",
    )

    
    plt.axis("equal")
  


    if plot_fake_vehicles is not None:
        legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
                           Line2D([0], [0], color="tab:orange", lw=4, label='Fake Measurements'),
                          Line2D([0], [0], color="blue", lw=4, label='Targets predicted', marker='^', markersize = 10, linestyle="None")]    

    elif gt is None:
        if type_ != 'GT':
            legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
                              Line2D([0], [0], color="blue", lw=4, label='Targets predicted', marker='^', markersize = 10, linestyle="None")] 
            
            # Temp for scenario visualization
#             legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
#                               Line2D([0], [0], color="blue", lw=4, label='Targets', marker='^', markersize = 10, linestyle="None")]  
        else:

            legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
                            Line2D([0], [0], color="blue", lw=4, label='Targets gt', marker='^', markersize = 10, linestyle="None")]   

            # If there are FPs
            if 1 in [1 for k in name_features if 'FP' in k]:
                legend_elements.append(Line2D([0], [0], color="black", lw=4, label='FP', marker='^', markersize = 10, linestyle="None"))

            # If we have positions of pedestrians
            if pedestrians_pos_xy is not None:
                legend_elements.append(Line2D([0], [0], color="black", lw=4, label='pedestrians pos', marker='x', markersize = 10, linestyle="None"))
             
                            
    else:
        legend_elements = [Line2D([0], [0], color="tab:gray", lw=4, label='Measurements'),
                          Line2D([0], [0], color="red", lw=4, label='Targets predicted != gt, gt pos', marker='^', markersize = 10, linestyle="None"),
                          Line2D([0], [0], color="blue", lw=4, label='Targets predicted != gt, predicted pos', marker='^', markersize = 10, linestyle="None"),
                          Line2D([0], [0], color="green", lw=4, label='Targets predicted = gt', marker='^', markersize = 10, linestyle="None")]

        
    legend1 = plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(+1, +1.31))#(+1, +1.16))

    elements = []
    for vehicle in vehicles:
        elements.append(Line2D([0], [0], color=colors[int(vehicle)], lw=4,  marker='s', markersize = 10, linestyle="None", label=f'Vehicle{vehicle}'))
        
    if plot_fake_vehicles is not None:  
        
        for i in range(len(fake_vehicles_instants)):
            elements.append(Line2D([0], [0], color='orange', lw=4,  marker='s', alpha=(i+1)/len(fake_vehicles_instants), markersize = 20, linestyle="None", label=f'Fake Vehicle time{fake_vehicles_instants[i]}'))

    legend2 = plt.legend(handles=elements, loc='upper left', ncol = 3, bbox_to_anchor=(-0.005, +1.31))#(-0.1, +1.16))
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    
    # Import Map
    img = plt.imread(image_path)
    # img = Image.open(image_path)
    y_shift = 0# +95 # 0 # +95
    x_shift = 0# -205 # 0#-205

    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = 100,100
    # Custom (rgb) grid color
    grid_color = [0,0,0,0]
    # Modify the image to include the grid
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    
    x_values = np.arange (-313, -100, (313-100)/9)
    y_values = np.arange (-14, 200, (200+14)/9)
    for x_ in x_values:
        for y_ in y_values:
            if (x_ == -313 or x_ == -100) and (x_!=-313 or y_!=200) and (x_!=-313 or y_!=-14) and (x_!=-100 or y_!=200) and (x_!=-100 or y_!=-14):
                plt.text(x_,y_,f'{y_:.1f}',color='k',ha='center',va='center')  
            if (y_ == -14 or y_ == 200) and (x_!=-313 or y_!=200) and (x_!=-313 or y_!=-14) and (x_!=-100 or y_!=200) and (x_!=-100 or y_!=-14):
                plt.text(x_,y_,f'{x_:.1f}',color='k',ha='center',va='center') 
                
    plt.text(-313,200,'(-313,200)',color='k',ha='center',va='center')
    plt.text(-313,-14,'(-313,-14)',color='k',ha='center',va='center')
    plt.text(-100,200,'(-100,200)',color='k',ha='center',va='center')
    plt.text(-100,-14,'(-100,-14)',color='k',ha='center',va='center')
    plt.imshow(img, extent=[-100+x_shift, -313+x_shift, -14+y_shift, 200+y_shift]) # from the edges of thr road 

    
    
    
    if title_savename is None:

        if gt is None:
            if type_ == 'GT':
                plt.title("Association vehicle-features GT")
                if save_pdf:
                    plt.savefig(DIR_EXPERIMENT + '/gt_association_vehicle_features.pdf', bbox_inches='tight')
                if save_eps:
                    plt.savefig(DIR_EXPERIMENT + '/gt_association_vehicle_features.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)
            else:
                plt.title("Association vehicle-features Prediction")
                if save_pdf:
                    plt.savefig(DIR_EXPERIMENT + '/prediction_association_vehicle_features.pdf', bbox_inches='tight')
                if save_eps:
                    plt.savefig(DIR_EXPERIMENT + '/prediction_association_vehicle_features.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)

                # Temp for scenario visualization
    #             plt.title("Association vehicle-features")
    #             plt.savefig(DIR_EXPERIMENT + '/association_vehicle_features.pdf', bbox_inches='tight')
    #             plt.savefig(DIR_EXPERIMENT + '/association_vehicle_features.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)

        elif plot_fake_vehicles is not None:  
            plt.title("Time Association vehicle-features")
            if save_pdf:
                plt.savefig(DIR_EXPERIMENT + '/time_association_vehicle_features.pdf', bbox_inches='tight') 
            if save_eps:
                plt.savefig(DIR_EXPERIMENT + '/time_association_vehicle_features.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)

        else:
            plt.title("Association vehicle-features")
            if save_pdf:
                plt.savefig(DIR_EXPERIMENT + '/association_vehicle_features.pdf', bbox_inches='tight')
            if save_eps:
                plt.savefig(DIR_EXPERIMENT + '/association_vehicle_features.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)

    else:
        plt.title(title_savename['title'])
        if save_pdf:
            plt.savefig(DIR_EXPERIMENT + f'/{title_savename["savename"]}.pdf', bbox_inches='tight')
        if save_jpg:
            plt.savefig(DIR_EXPERIMENT + f'/{title_savename["savename"]}.jpg', bbox_inches='tight')
        if save_eps:
            plt.savefig(DIR_EXPERIMENT + f'/{title_savename["savename"]}.eps', bbox_inches='tight', format='eps', rasterized=True,transparent=True)
        
    plt.rcParams.update({'font.size': 22})  
   
    return colors















































































































































































































































































































































































































































































































