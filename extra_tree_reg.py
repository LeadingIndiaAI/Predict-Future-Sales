import numpy as np 
import pandas as pd
from sklearn import *


test  = pd.read_csv('test.csv').set_index('ID')


data=pd.read_csv('data.csv')
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)



model = ensemble.ExtraTreesRegressor(
    max_depth=20,
    random_state=42,
    n_estimators=200,   
    n_jobs=-1)

model.fit(
    X_train, 
    Y_train)


Y_test =model.predict(X_test.values).clip(0, 20)

submission = pd.DataFrame({
     "ID": test.index,
     "item_cnt_month": Y_test
})

submission.to_csv('ext_tree.csv', index=False)

