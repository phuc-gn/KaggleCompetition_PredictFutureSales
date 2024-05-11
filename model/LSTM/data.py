import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


def preprocess(path):
    data = pd.read_csv(path)
    data['item_cnt_day'] = data['item_cnt_day'].clip(lower = 0)
    data = data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
    data.rename(columns = {'item_cnt_day': 'item_cnt_month'}, inplace = True)
    data = data.pivot_table(index = ['shop_id', 'item_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value = 0)
    data = data.loc[(data > 0).sum(axis = 1) > 0]

    features = pd.DataFrame()
    features['median'] = data.median(axis = 1)
    features['std'] = data.std(axis = 1)
    features['max'] = data.max(axis = 1)
    features['min'] = data.min(axis = 1)
    features['skew'] = data.skew(axis = 1)
    features['iqr'] = data.quantile(0.75, axis = 1) - data.quantile(0.25, axis = 1)

    data = pd.concat([features, data], axis = 1)

    return data.values


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
    def __init__(self, path):
        self.data = torch.tensor(preprocess(path), dtype = torch.float32)

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