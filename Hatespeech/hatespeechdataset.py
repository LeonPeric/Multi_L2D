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


class HatespeechDataset(Dataset):
    def __init__(self, data_path='hatespeech_data', split='train'):
        super(HatespeechDataset).__init__()
        data = load_data(data_path)
        print(data['test'].keys())
        if split == 'train':
            self.X = torch.from_numpy(data['X']).float()
            self.Y = torch.from_numpy(data['Y']).long()
            self.hlabel = data['hpred']

            self.X = self.X[self.Y<2]
            self.hlabel = self.hlabel[self.Y<2]
            self.Y = self.Y[self.Y<2]

            
        else:
            self.X = torch.from_numpy(data[split]['X']).float()
            self.Y = torch.from_numpy(data[split]['Y']).long()
            self.hlabel = data[split]['hpred']

            self.X = self.X[self.Y<2]
            self.hlabel = self.hlabel[self.Y<2]
            self.Y = self.Y[self.Y<2]


    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.hlabel[index]

    def __len__(self):
        return self.X.shape[0]


if __name__ == "__main__":
    split = HatespeechDataset()
    dl = DataLoader(split, batch_size=1024, shuffle=True)
    for batch in dl:
        X, Y, H = batch
        print(X.shape, Y.shape, H.shape)
        break
