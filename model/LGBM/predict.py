import joblib
import pandas as pd

raw_data = pd.read_csv('./sales_train.csv')

monthly_sales = raw_data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum().reset_index()
monthly_sales.rename(columns={'item_cnt_day': 'item_cnt_month'}, inplace=True)
monthly_sales_matrix = monthly_sales.pivot_table(index = ['shop_id', 'item_id'], columns = 'date_block_num', values = 'item_cnt_month', fill_value=0)

features = monthly_sales_matrix.columns[:-1]
target = monthly_sales_matrix.columns[-1]

# model loading
with open('model.joblib', 'rb') as f:
    model = joblib.load(f)

print(model.get_params())
# prediction
test_data = pd.read_csv('./test.csv')

test_df = test_data.join(monthly_sales_matrix, on = ['shop_id', 'item_id'], how = 'left')
test_df.fillna(0, inplace = True)

predictions = model.predict(test_df[features])
predictions = pd.Series(predictions).clip(0, 20)
submit_data = pd.concat([test_data['ID'], pd.Series(predictions, name = 'item_cnt_month')], axis = 1)

submit_data.to_csv('submission.csv', index = False)