import argparse

from tqdm import tqdm
import pandas as pd

import torch

from model import SalesLSTM as model_, config_lstm as config_
from trainer import load_checkpoint
from utils import set_device

tqdm.pandas()


def prepare_data(data_path, test_path):
    data_df = pd.read_csv(data_path)
    test_df = pd.read_csv(test_path)

    data_df['item_cnt_day'] = data_df['item_cnt_day'].clip(lower = 0)
    data_df = data_df.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
    data_df.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace = True)
    data_df = data_df.pivot_table(index = ['shop_id', 'item_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value=0)

    test_df = test_df.join(data_df, on = ['shop_id', 'item_id'], how = 'left')
    ID_col = test_df['ID'].copy()
    test_df = test_df.drop(columns = ['shop_id', 'item_id', 'ID'])
    test_df = test_df.iloc[:, 1:]

    features = pd.DataFrame()
    features['median'] = test_df.median(axis = 1)
    features['std'] = test_df.std(axis = 1)
    features['max'] = test_df.max(axis = 1)
    features['min'] = test_df.min(axis = 1)
    features['skew'] = test_df.skew(axis = 1)
    features['iqr'] = test_df.quantile(0.75, axis = 1) - test_df.quantile(0.25, axis = 1)

    test_df = pd.concat([features, test_df], axis = 1)

    return test_df, ID_col


def predict(model, data):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(data.values, dtype = torch.float32).view(1, -1)).numpy()[0]


def main():
    parser = argparse.ArgumentParser(description = 'Predict sales')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to predict on')
    args = parser.parse_args()

    device = set_device(args.device)

    data_path = '../../data/sales_train.csv'
    test_path = '../../data/test.csv'

    test_df, ID_col = prepare_data(data_path, test_path)

    # Load model
    model = model_(**config_())
    with open('checkpoint/model.pth', 'rb') as f:
        model = load_checkpoint(f, model, device)

    # Predict
    predictions = test_df.progress_apply(lambda x: predict(model, x), axis = 1)
    predictions = predictions.clip(0, 20).fillna(0)

    # Save predictions
    submit_data = pd.concat([ID_col, pd.DataFrame(predictions, columns = ['item_cnt_month'])], axis = 1)
    submit_data.to_csv('../../submissions/submission.csv', index = False)


if __name__ == '__main__':
    main()