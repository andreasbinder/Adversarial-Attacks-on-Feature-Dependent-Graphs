import os.path as osp

from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, download_url, InMemoryDataset
from torch_geometric.data import Data

import torch
from torch_geometric.nn import radius_graph

# from data import get_dataset

from src.adversarial_attacks_on_feature_dependent_graphs.data import get_dataset

import sys
sys.path.append('/nfs/homedirs/bindera/Adversarial Attacks on Feature-Dependent GraphsAdversarial Attacks on Feature-Dependent Graphs/src/adversarial_attacks_on_feature_dependent_graphs')

class PointCloudDataset(InMemoryDataset):  # Dataset
    def __init__(self, data_folder = '../../../shared/modelnet/modelnet40_normal_resampled', split = 'train', radius=1, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(None, transform, pre_transform, pre_filter)

        subset = self.load(data_folder=data_folder, split=split)

        self.data_list = self.extract_data_from_dict(subset, radius)

        
    def load(self, data_folder, split):
        
        data_train, data_val, data_test = get_dataset(name = 'modelnet40', data_folder = data_folder)

        if split == 'train':
            return data_train
        elif split == 'val':
            return data_val
        elif split == 'test':
            return data_test
        else:
            raise KeyError

    def to_tensor(self, dataset):
        xs , ys = [], []

        for x, y in dataset:
            xs.append(torch.tensor(x))
            ys.append(torch.tensor(y))
        
        return torch.stack(xs), torch.stack(ys)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        return data

    def extract_data_from_dict(self, subset, radius):
        
        # subset = self.to_tensor(subset)

        list_of_data_objects = []

        for x, y in tqdm(subset):
          
            x = torch.tensor(x)
            
            edge_index = radius_graph(x=x, r=radius)
            data = Data(x = x, 
                        edge_index = edge_index, # .t().contiguous()
                        y = torch.tensor([y], dtype=torch.long))

            list_of_data_objects.append(data)

        return list_of_data_objects