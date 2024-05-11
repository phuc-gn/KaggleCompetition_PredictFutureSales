import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, root_mean_squared_error

from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV

# read data
raw_data = pd.read_csv('./sales_train.csv')

monthly_sales = raw_data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
monthly_sales_matrix = monthly_sales.pivot_table(index = ['shop_id', 'item_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value=0)

features = monthly_sales_matrix.columns[:-1]
target = monthly_sales_matrix.columns[-1]

model = Pipeline([
    ('regressor', LGBMRegressor())
])


param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.005, 0.01, 0.1, 0.2],
    'regressor__max_depth': [5, 10, 15],
    'regressor__num_leaves': [20, 30, 40, 50, 100, 200, 250],
    'regressor__feature_fraction': [0.6, 0.75, 0.9]
    # 'regressor__min_child_samples': [20, 30, 40],
    # 'regressor__subsample': [0.8, 0.9, 1],
    # 'regressor__colsample_bytree': [0.8, 0.9, 1]
}

# tscv = TimeSeriesSplit(n_splits = 3)
# grid_search = GridSearchCV(model, param_grid, scoring = 'neg_root_mean_squared_error', cv = tscv, verbose = 4)

# # fit model
# grid_search.fit(monthly_sales_matrix[features], monthly_sales_matrix[target])
# model = grid_search.best_estimator_
param_grid = ParameterGrid(param_grid)

best_score = np.inf
best_model = None
for params in param_grid:
    # Set the parameters and fit the model
    model.set_params(**params)
    model.fit(monthly_sales_matrix[features], monthly_sales_matrix[target])
    
    # Predict on the training set and calculate the error
    predictions = model.predict(monthly_sales_matrix[features])
    score = root_mean_squared_error(monthly_sales_matrix[target], predictions)
    print(f'Score for parameters {params}: {score}')
    # If the score is better than the best_score, update best_score and best_model
    if score < best_score:
        best_score = score
        best_model = model

model = best_model
# save model
with open('model.joblib', 'wb') as f:
    joblib.dump(model, f)