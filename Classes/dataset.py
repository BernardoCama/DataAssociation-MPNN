import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import random
from torch_geometric.data import InMemoryDataset
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

def random_flip_xy(raw_data):
    # Make a deep copy of the data to avoid modifying the input
    flipped_data = deepcopy(raw_data)

    # Iterate over all vehicles and timesteps
    for vehicle in range(flipped_data.shape[0]):
        for timestep in range(flipped_data.shape[1]):
            # Skip entries where the 'boxes' field is empty
            if flipped_data[vehicle][timestep][0][0]['boxes'].shape[-1] > 0 and flipped_data[vehicle][timestep][0][0]['boxes'].shape[0] > 0:
                # Randomly choose between x (0) and y (1)
                flip_axis = random.choice([0, 1])
                # Flip the chosen axis
                flipped_data[vehicle][timestep][0][0]['boxes'][flip_axis, :] *= -1

    concat_data = np.concatenate([raw_data, flipped_data], axis=1)
    return concat_data

class GraphDataset(InMemoryDataset):
    def __init__(self, root, raw_data=None, dir_dataset=None, number_instants=None, number_vehicles=None, connectivity_matrix_dict=None, name_processed_dataset=None, transform=None, pre_transform=None, augment = 0):
        
        self.raw_data = raw_data
        self.dir_dataset = dir_dataset
        self.number_instants = number_instants
        self.number_vehicles = number_vehicles
        self.connectivity_matrix_dict = connectivity_matrix_dict
        if augment:
            self.raw_data = random_flip_xy(raw_data)
            self.number_instants *= 2

        self.name_processed_dataset = name_processed_dataset
                
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    # Where to put output and its name
    @property
    def processed_file_names(self):
        if self.name_processed_dataset is None:
            return [self.dir_dataset + 'GraphDataset.dataset']
        else:
            return [self.dir_dataset + f'{self.name_processed_dataset}.dataset']

    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        
        for instant in tqdm(range(self.number_instants)):
        
            edge_index = []   # indexes of edges
            edge_labels = []  # labels of edges
            edge_attr = []    # attributes of edges
            source_nodes = []
            dest_nodes = []
            x = {}            
            node_id = {}      # id at the node
            feature_name = {} # name of the feature
            track_id = {}     # identificative of the tracking
            node_labels = {}

            node = 0

            for vehicle in range(self.number_vehicles):

                feature = 0

                for name in self.connectivity_matrix_dict[instant][vehicle]:

                    box = self.raw_data[vehicle][instant][0][0][0].reshape([3,8,-1])[:,:,feature]
                    
                    centroid = np.mean(box, 1)
                    
                    add_source_node = 0

                    for vehicle2 in range(vehicle+1, self.number_vehicles):

                        feature2 = 0

                        for name2 in self.connectivity_matrix_dict[instant][vehicle2]:

                            box2 = self.raw_data[vehicle2][instant][0][0][0].reshape([3,8,-1])[:,:,feature2]
                            
                            centroid2 = np.mean(box2, 1)
                            
                            if np.linalg.norm(centroid-centroid2) < 10:  # m
                            
                                add_source_node = 1

                                node2 = self.connectivity_matrix_dict[instant][vehicle2][name2]
                                
                                x[node] = [box.flatten().tolist(), vehicle]
                                x[node2] = [box2.flatten().tolist(), vehicle2]

                                node_labels[node] = 1 if 'FP' in name else 0
                                node_labels[node2] = 1 if 'FP' in name2 else 0

                                node_id[node] = node
                                node_id[node2] = node2 
                                
                                feature_name[node] = name
                                feature_name[node2] = name2   
                                
                                track_id[node] = name
                                track_id[node2] = name2      
                                
                                source_nodes.append(node)

                                dest_nodes.append(node2)

                                edge_labels.append(1 if (name == name2) and (('FP' not in name) and ('FP' not in name2)) else 0)

                                edge_attr.append([box[0,0]-box2[0,0],
                                                  box[1,0]-box2[1,0], 
                                                  box[2,0]-box2[2,0],
                                                  box[0,7]-box2[0,7],
                                                  box[1,7]-box2[1,7],
                                                  box[2,7]-box2[2,7]])

                            feature2 += 1                            
                        
                    node += 1
                    
                    feature += 1

            # Node features
            x = OrderedDict(sorted(x.items()))
            # Node vehicle (correspondence node_ith - vehicle, order of nodes is the same of x)
            node_vehicle = torch.tensor([v[1] for k, v in x.items()], dtype=torch.float)  
            x = torch.tensor([v[0] for k, v in x.items()], dtype=torch.float)
            # Node labels
            node_labels = OrderedDict(sorted(node_labels.items()))
            node_labels = torch.tensor([v for k, v in node_labels.items()], dtype=torch.long)
            # Edge indexes
            encoder = LabelEncoder().fit(source_nodes + dest_nodes)
            edge_index = torch.tensor([encoder.transform(source_nodes).tolist(), encoder.transform(dest_nodes).tolist()], dtype=torch.long)
            # Edge features
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            # Edge labels
            edge_labels = torch.tensor(edge_labels, dtype=torch.long)
            # Time instant
            time_instant = torch.tensor(instant, dtype=torch.long)
            # Id of each node
            node_id = torch.tensor(encoder.transform([v for k, v in node_id.items()]), dtype=torch.float)
            # Name of each node
            feature_name = OrderedDict(sorted(feature_name.items()))
            feature_name = tuple([v for k, v in feature_name.items()])
            # Id for tracking data association for each node
            track_id = OrderedDict(sorted(track_id.items()))
            track_id = tuple([v for k, v in track_id.items()])

            data = Data (x = x,
                        edge_attr = torch.cat((edge_attr, edge_attr), dim = 0),
                        edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]))), dim=1))
            data.node_labels = node_labels
            data.edge_labels = torch.cat((edge_labels, edge_labels), dim = 0)
            data.time_instant = time_instant
            data.node_vehicle = node_vehicle
            data.node_id = node_id
            data.feature_name = feature_name
            data.track_id = track_id

            if data.num_nodes != 0:
                data_list.append(data)
        
        print(len(data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




























































































































































































































































































































































































































































































































































