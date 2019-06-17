import numpy as np 
import pandas as pd
import tensorflow as tf


test  = pd.read_csv('data/test.csv').set_index('ID')


data=pd.read_csv('data.csv')
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

model=tf.keras.Sequential({
    tf.keras.layers.Dense(20,input_shape=(34,),activation='relu'),
    tf.keras.layers.normalization.BatchNormalization(),
    tf.keras.layers.Dense(15,activation='relu'),
    tf.keras.layers.Dropout(rate=0.20),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.normalization.BatchNormalization(),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1)
})


model.compile('adam',loss=tf.losses.mean_squared_error,metrics=['mse'])

model.fit(X_train.values,Y_train.values,epochs=100,validation_data=(X_valid.values,Y_valid.values),batch_size=15)

Y_test =model.predict(X_test.values).clip(0, 20)

submission = pd.DataFrame({
     "ID": test.index,
     "item_cnt_month": Y_test
})

submission.to_csv('nn_sub.csv', index=False)




