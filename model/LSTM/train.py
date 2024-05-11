import os
import argparse
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Sales
from model import SalesLSTM as model_, config_lstm as config_
from trainer import train, checkpoint
from utils import set_device


def main():
    parser = argparse.ArgumentParser(description = 'Model training')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Number of epochs to train the model')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Learning rate for the optimiser')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size for the dataloader')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to train the model on')
    args = parser.parse_args()

    device = set_device(args.device)

    data_path = '../../data/sales_train.csv'

    data = Sales(data_path)

    train_dataloader = DataLoader(data,
                                  batch_size = args.batch_size,
                                  shuffle = True,
                                  pin_memory = True)

    model = model_(**config_()).to(device)
    criteria = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size = 2, gamma = 0.1)

    training_loss = []

    time_start = timeit.default_timer()

    for epoch in range(args.epochs):
        train(model, criteria, train_dataloader, optimiser, device, epoch, training_loss)
        scheduler.step()

    time_end = timeit.default_timer()

    print(f'\nTraining took {time_end - time_start} seconds')

    checkpoint('checkpoint/model.pth', model)


if __name__ == '__main__':
    main()