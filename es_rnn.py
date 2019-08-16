import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict, deque
from IPython.display import Image
# %matplotlib inline
from common.es import ES
from common.denormalization import Denormalization
from common.utils import load_data, mape, TimeSeriesTensor, create_evaluation_df

pd.options.display.float_format = '{:,.2f}'.format
np.set_printoptions(precision=2)
warnings.filterwarnings("ignore")

energy = load_data('data/')
energy.head()

valid_start_dt = '2014-08-30 08:00:00'
test_start_dt = '2014-10-31 11:00:00'
test_end_dt = '2014-12-30 18:00:00'

T = 6
HORIZON = 3


train = energy.copy()[energy.index < valid_start_dt][['load']]

from sklearn.preprocessing import MinMaxScaler
# transforming data
scaler = MinMaxScaler()
scaler.fit(train[['load']])
train[['load']] = scaler.transform(train)

tensor_structure = {'X':(range(-T+1, 1), ['load'])}
train_inputs = TimeSeriesTensor(train, 'load', HORIZON, tensor_structure)

print("y_train shape: ", train_inputs['target'].shape)
print("x_train shape: ", train_inputs['X'].shape)

# similarly for validation set
look_back_dt = dt.datetime.strptime(valid_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
valid = energy.copy()[(energy.index >=look_back_dt) & (energy.index < test_start_dt)][['load']]
valid[['load']] = scaler.transform(valid)
valid_inputs = TimeSeriesTensor(valid, 'load', HORIZON, tensor_structure)

## build model

from keras.models import Model
from keras.layers import Input, GRU, Dense, Lambda
from keras.callbacks import EarlyStopping

LATENT_DIM = 5 # number of units in the RNN layer
BATCH_SIZE = 48 # number of samples per mini-batch
EPOCHS = 10 # maximum number of times the training algorithm will cycle through all samples
m = 24 # seasonality length


model_input = Input(shape=(None, 1))
[normalized_input, denormalization_coeff] = ES(HORIZON, m, BATCH_SIZE, T)(model_input)
gru_out = GRU(LATENT_DIM)(normalized_input)
model_output_normalized = Dense(HORIZON)(gru_out)
model_output = Denormalization()([model_output_normalized, denormalization_coeff])
model = Model(inputs=model_input, outputs=model_output)

model.compile(optimizer='RMSprop', loss='mse')

model.summary()

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20)
model.fit(train_inputs['X'],
          train_inputs['target'],
          batch_size=BATCH_SIZE,
          shuffle=False,
          epochs=EPOCHS,
          validation_data=(valid_inputs['X'], valid_inputs['target']),
          callbacks=[earlystop],
          verbose=1)

## evaluate model
look_back_dt = dt.datetime.strptime(test_start_dt, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=T-1)
test = energy.copy()[test_start_dt:test_end_dt][['load']]
test[['load']] = scaler.transform(test)
test_inputs = TimeSeriesTensor(test, 'load', HORIZON, tensor_structure)
predictions = model.predict(test_inputs['X'], batch_size=BATCH_SIZE)

eval_df = create_evaluation_df(predictions, test_inputs, HORIZON, scaler)
eval_df.head()

eval_df['APE'] = (eval_df['prediction'] - eval_df['actual']).abs() / eval_df['actual']

print(eval_df.groupby('h')['APE'].mean())

print("mape:", mape(eval_df['prediction'], eval_df['actual']))

plot_df = eval_df[(eval_df.timestamp<'2014-11-08') & (eval_df.h=='t+1')][['timestamp', 'actual']]
for t in range(1, HORIZON+1):
    plot_df['t+'+str(t)] = eval_df[(eval_df.timestamp<'2014-11-08') & (eval_df.h=='t+'+str(t))]['prediction'].values

fig = plt.figure(figsize=(15, 8))
ax = plt.plot(plot_df['timestamp'], plot_df['actual'], color='red', linewidth=4.0)
ax = fig.add_subplot(111)
ax.plot(plot_df['timestamp'], plot_df['t+1'], color='blue', linewidth=4.0, alpha=0.75)
ax.plot(plot_df['timestamp'], plot_df['t+2'], color='blue', linewidth=3.0, alpha=0.5)
ax.plot(plot_df['timestamp'], plot_df['t+3'], color='blue', linewidth=2.0, alpha=0.25)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
ax.legend(loc='best')
plt.show()

