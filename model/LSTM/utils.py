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