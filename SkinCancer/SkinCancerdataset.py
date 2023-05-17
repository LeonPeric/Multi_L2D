from __future__ import division

import os
import random

import numpy as np
import pickle5 as pickle
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def load_data(file_name):
    assert(os.path.exists(file_name+'.pkl'))
    with open(file_name + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

class SkinCancerDataset(Dataset):
    def __init__(self, data_path='SkinCancer', split='train', error_rate=0.0):
        super(SkinCancerDataset).__init__()
        data = load_data(data_path)
        if split == 'train':
            self.X = torch.from_numpy(data['X']).float()
            self.Y = torch.from_numpy(data['Y']).long()
            self.hlabel = data["Y"]
            self.Y[:] = torch.from_numpy(np.where((np.random.rand(*self.Y.shape) < error_rate), 1-self.Y, self.Y))
            # self.hconf = data['hconf']
        else:
            self.X = torch.from_numpy(data[split]['X']).float()
            self.Y = torch.from_numpy(data[split]['Y']).long()
            self.hlabel = data["Y"]
            self.Y[:] = torch.from_numpy(np.where((np.random.rand(*self.Y.shape) < error_rate), 1-self.Y, self.Y))

    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.hlabel[index]

    def __len__(self):
        return self.X.shape[0]
    

if __name__ == "__main__":
    split = SkinCancerDataset()
    dl = DataLoader(split, batch_size=1024, shuffle=True)
    for batch in dl:
        X, Y, H = batch
        print(X.shape, Y.shape, H.shape)