# IMPORT REQUIRED MODULES

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM ,Dropout
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn import metrics
from scipy import stats

# TAKE THE INPUT FOR THE PROGRAM

token = str(input('Enter the token - '))
interval = str(input('Enter the interval Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo] - '))
date = str(input('Enter the start date in YYYY-MM-DD format - '))

# LOAD THE DATA 
df = yf.download(token ,interval=interval,start_date=date)
while df.shape[0] == 0:
    token = str(input('Enter the token - '))
    interval = str(input('Enter the interval Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo] - '))
    date = str(input('Enter the start date in YYYY-MM-DD format - '))
    df = yf.download(token ,interval=interval,start=date)

# CREATE ARTIFICIAL VARIABLES 

for i in range(20,150,10):
    df[f'SMA_{i}'] = ta.SMA(df['Close'] , i)
    df[f'EMA_{i}'] = ta.EMA(df['Close'], i)
    
for i in range(7,150,7):
    df[f'RSI_{i}'] = ta.RSI(df['Close'],i)
    df[f'avg_{i}'] = ta.ADX(df['High'],df['Low'], df['Close'], timeperiod=i)

for i in range(5,100,5):
    df[f'linear_reg_{i}'] = ta.LINEARREG(df['Close'],i)
    df[f'up_band_{i}'], df[f'mid_band_{i}'], df[f'low_band_{i}'] = ta.BBANDS(df['Close'], timeperiod =i)

for i in range(72):
    df[f'O_{i+1}'] = df['Open'].shift((i+1))
    df[f'C_{i+1}'] = df['Close'].shift(i+1)
    df[f'L_{i+1}'] = df['Low'].shift(i+1)
    df[f'H_{i+1}'] = df['High'].shift(i+1)

    df[f'O/C_{i+1}'] = df['Open'] / df['Close']
    df[f'H/L_{i+1}'] = df['High'] / df['Low']

for i in range(3):
    df[f'to_predict_{i+1}'] = df['Close'].shift(-(i+1))
    
# FILTER THE ARTIFICIAL VARIABLES USING P_VALUE 

x_col = list(df.columns)[:-3]
y_col = list(df.columns)[-3:]
df.dropna(inplace=True)

p_val = []
for col in df[x_col]:
    r, p = stats.pearsonr(df['to_predict_1'] , df[col])
    p_val.append((col,round(p,5)))
x = []
X_col = []
for i in p_val:
    if i[1]<0.05:
        X_col.append(i[0])
        x.append(i)

# SPLIT THE DATA TO TRAIN THE MODEL 

X_train =  df[X_col].iloc[:int(df.shape[0]*.95)] 
X_test = df[X_col].iloc[int(df.shape[0]*.95):] 
y_train = df[y_col].iloc[:int(df.shape[0]*.95)] 
y_test =  df[y_col].iloc[int(df.shape[0]*.95): ]

# USING KERAS TUNER TO GENERATE OPTIMAL NUERAL NETWORK


def model_builder(hp):
    
    model = Sequential()
    
    hp_activation = hp.Choice('activation', values=['relu', 'tanh'])
    hp_layer_0 = hp.Int('layer_0', min_value=64, max_value=1024, step=128)
    hp_layer_1 = hp.Int('layer_1', min_value=64, max_value=2048, step=256)
    hp_layer_2 = hp.Int('layer_2', min_value=64, max_value=2048, step=256)
    hp_layer_3 = hp.Int('layer_3', min_value=64, max_value=2048, step=256)
    hp_layer_4 = hp.Int('layer_4', min_value=64, max_value=512, step=64)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
# LSTM LAYER
    model.add(LSTM(units=hp_layer_0 , input_shape=(X_train.shape[1],1)))
    
# FIRST DENSE LAYER
    model.add(Dense(units=hp_layer_1, activation=hp_activation))
    model.add(
            Dropout(rate=hp.Float(
                'dropout_0',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
# SECOND DENSE LAYER   
    model.add(Dense(units=hp_layer_2, activation=hp_activation))
    model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
# THIRD DENSE LAYER
    model.add(Dense(units=hp_layer_3, activation=hp_activation))
    model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
# FORTH DENSE LAYER
    model.add(Dense(units=hp_layer_4, activation=hp_activation))
    model.add(
            Dropout(rate=hp.Float(
                'dropout_3',
                min_value=0.0,
                max_value=0.1,
                default=0.0,
                step=0.05,
            ))
        )
# OUTPUT LAYER
    model.add(Dense(y_train.shape[1]))

# COMPILE THE MODEL
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanSquaredError())
    
    return model

# KERAS TUNER
tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=150,
                     factor=6,
                     directory='dir',
                     project_name='x')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(X_train, y_train, epochs=150, validation_data=(X_test,y_test), callbacks=[stop_early])

print('----------------------------------------------------------')

# GET THE BEST MODEL AND TRAIN IT AGAIN WITH HIGHER EPOCH WITH CALLBACK TO SAVE TIME
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=150, validation_split=0.2,
                    callbacks=[stop_early])

# SAVE THE MODEL
model.save(f'{token}_model.h5')
with open(f'{token}_features.txt', 'wb') as f:
    f.write(','.join(x for x in X_col))
    f.write(','.join(y for x in y_col))
    
# PRINT THE COMPARISON MATRICS 
y_pred = model.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# PREDICT NEW DATA 
file = str(input('Enter the file name of the new data - '))
X_pred = pd.read_csv(file)
y_pred = model.predict(X_pred)
plt.figure(figsize=(15,9))
sns.lineplot([i for i in range(len(y_pred))] , y_pred)
plt.show()