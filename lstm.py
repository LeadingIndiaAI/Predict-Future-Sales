import numpy as np 
import pandas as pd
from xgboost import XGBRegressor

test  = pd.read_csv('data/test.csv').set_index('ID')

data=pd.read_csv('data.csv')

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)


x_train_reshaped = X_train.values.reshape((X_train.values.shape[0], 1, X_train.values.shape[1]))
x_val_resaped = X_valid.values.reshape((X_valid.values.shape[0], 1, X_valid.values.shape[1]))


model=keras.Sequential([
    keras.layers.CuDNNLSTM(units=8,input_shape=(1,38)),
    keras.layers.Dense(units=12, activation='relu'),
    keras.layers.Dense(12,activation='relu'),
    keras.layers.Dropout(rate=0.20),
    keras.layers.Dense(8,activation='relu'),
    keras.layers.Dropout(rate=0.10),
    keras.layers.Dense(5,activation='relu'),
    keras.layers.Dense(5,activation='relu'),
    keras.layers.Dense(1,activation='linear')
])

Y_test =model.predict(X_test.values).clip(0, 20)

submission = pd.DataFrame({
     "ID": test.index,
     "item_cnt_month": Y_test.ravel()
})

submission.to_csv('lstm_sub.csv', index=False)

