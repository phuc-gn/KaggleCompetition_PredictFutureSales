import pandas as pd

import torch
import torch.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device):
    if device == 'cuda' and torch.cuda.is_available():
        print('Using GPU')
        device = torch.device('cuda')
    else:
        print('Using CPU')
        device = torch.device('cpu')

    return device

def preprocess(path):
    data = pd.read_csv(path)
    data['item_cnt_day'] = data['item_cnt_day'].clip(lower = 0)
    data = data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
    data.rename(columns = {'item_cnt_day': 'item_cnt_month'}, inplace = True)
    data = data.pivot_table(index = ['shop_id', 'item_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value = 0)
    data = data.loc[(data > 0).sum(axis = 1) >= 1]

    return data.values