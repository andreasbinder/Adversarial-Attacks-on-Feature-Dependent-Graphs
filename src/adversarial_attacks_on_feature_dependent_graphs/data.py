import os
import sys

import h5py
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.utils import _log_api_usage_once



# sys.path.append('../../../reference_implementations/Pointnet_Pointnet2_pytorch/data_utils')
sys.path.append('/nfs/homedirs/bindera/reference_implementations/Pointnet_Pointnet2_pytorch/data_utils')

# print(sys.path)

from ModelNetDataLoader import ModelNetDataLoader

#from pointcloud_invariance_smoothing.utils import dotdict
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ToPointCloud(nn.Module):
    def __init__(self, threshold=0.5, n_points=256):
        super(ToPointCloud, self).__init__()
        _log_api_usage_once(self)

        self.threshold = 0.5
        self.n_points = n_points

    def forward(self, image: torch.Tensor):
        if not torch.all((image <= 1) & (image >= 0)):
            raise ValueError('Only support pixel values in [0, 1]')

        assert image.ndim == 3 and image.shape[0] == 1
        image = image[0]

        height, width = image.shape
        coordinates = torch.torch.stack(torch.meshgrid(
                                            torch.linspace(0, 1, height),
                                            torch.linspace(0, 2, width)))

        threshold_mask = image >= self.threshold
        point_cloud = coordinates[:, threshold_mask].T  # N_points x 2

        # Normalize
        point_cloud -= point_cloud.mean(dim=0)
        max_len = torch.linalg.norm(point_cloud, dim=1).max()
        if max_len > 0:
            point_cloud /= max_len

        if len(point_cloud) < self.n_points:
            padding_value = point_cloud[0].unsqueeze(0)  # PointNet does max-pooling

            point_cloud = torch.cat(
                [point_cloud,
                    torch.repeat_interleave(padding_value, self.n_points - len(point_cloud), dim=0)
                 ],
                dim=0)

        elif len(point_cloud) > self.n_points:
            subsample_idcs = torch.randperm(len(point_cloud))[:self.n_points]
            point_cloud = point_cloud[subsample_idcs]

        return point_cloud


class ScanObjectNN(Dataset):
    def __init__(self, root, split='train', pca_preprocessed=True):
        self.data_list = []
        self.split = split

        if split not in ['train', 'test']:
            raise ValueError('Only have train and test set')

        data_dir = os.path.join(root,
                                'pca' if pca_preprocessed else 'ori',
                                split)

        for file_name in open(os.path.join(data_dir, f'{split}_list.txt')):
            self.data_list.append(os.path.join(data_dir, file_name).rstrip())

    def __getitem__(self, ind):
        file = h5py.File(self.data_list[ind] + '.h5', 'r', swmr=True)
        data = file['data'][:]
        pointcloud = data[:1024, :]
        data, label = torch.from_numpy(pointcloud), torch.from_numpy(file['label'][:])
        file.close()
        return data, int(label)

    def __len__(self):
        return len(self.data_list)


class RemapTargets(nn.Module):
    def __init__(self, remap_dict): # : dict[int, int]
        super(RemapTargets, self).__init__()
        _log_api_usage_once(self)

        self.remap_dict = remap_dict

    def forward(self, x):
        return self.remap_dict[x]


def get_dataset(name: str, data_folder: str, val_percentage: float = 0.2):
    name = name.lower()
    assert name in ['modelnet40', 'mnist', 'scanobjectnn']

    if name == 'modelnet40':

        data_args = {
            'num_point': 1024,
            'use_uniform_sample': True,
            'use_normals': False,
            'num_category': 40
        }

        data_args = dotdict(data_args)

        data_train = ModelNetDataLoader(root=data_folder, args=data_args,
                                        split='train', process_data=True)

        data_test = ModelNetDataLoader(root=data_folder, args=data_args,
                                       split='test', process_data=True)

    elif name == 'mnist':

        target_transform = RemapTargets({
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: 5,  6: 6, 7: 7, 8: 8, 9: 6
        })

        data_train = MNIST(data_folder, train=True,
                           transform=T.Compose([T.ToTensor(), ToPointCloud()]),
                           target_transform=target_transform)

        data_test = MNIST(data_folder, train=False,
                          transform=T.Compose([T.ToTensor(), ToPointCloud()]),
                          target_transform=target_transform)

    elif name == 'scanobjectnn':
        data_train = ScanObjectNN(root=data_folder, split='train')
        data_test = ScanObjectNN(root=data_folder, split='test')

    n_val = int(val_percentage * len(data_train))
    n_train = len(data_train) - n_val

    data_train, data_val = random_split(data_train, (n_train, n_val),
                                        generator=torch.Generator().manual_seed(0))

    return data_train, data_val, data_test
