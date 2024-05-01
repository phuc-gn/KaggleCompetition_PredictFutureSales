import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

class Sales(Dataset):
    """
    A custom dataset class for sales data.

    Args:
        path (str): The path to the sales data file.

    Attributes:
        data (torch.Tensor): The preprocessed sales data stored as a tensor.

    Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Returns a specific item from the dataset.

    """
    def __init__(self, data, train = True):
        if train:
            self.data = torch.tensor(data[:, :-1], dtype = torch.float32)
        else:
            self.data = torch.tensor(data[:, 1:], dtype = torch.float32)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.

        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a specific item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the features and the target of the item.

        """
        return self.data[idx][:-1], self.data[idx][-1]