import numpy as np 
import pandas as pd
from xgboost import XGBRegressor

test  = pd.read_csv('test.csv').set_index('ID')

data=pd.read_csv('data.csv')
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


model = XGBRegressor(
    max_depth=12,
    gamma=2,
    n_estimators=5000,
    min_child_weight=1,
    objective = 'reg:linear',
    colsample_bytree=0.5,
    subsample=0.8,
    reg_alpha=1,
    eta=0.05,
    seed=42)

model.fit(
    X_train,
    Y_train,
    eval_metric="rmse",
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds = 10)

Y_test =model.predict(X_test.values).clip(0, 20)

submission = pd.DataFrame({
     "ID": test.index,
     "item_cnt_month": Y_test.ravel()
})

submission.to_csv('xgb_sub.csv', index=False)

