import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def getIndexTest(array):
    ind = []
    for i in range(array.shape[0]):
        if array[i, 4, 50]>=0.1062 and array[i,4,50]<=0.16: #10 a 15 min
            ind.append(i)
        #print(array[i, 4,:])
    return ind

def getIndexTrain(array):
    ind = []
    for i in range(array.shape[0]):
        if array[i, 4, 50]>=0.1062 and array[i,4,50]<=0.16: #10 a 15 min
            ind.append(i)
    return ind

def inicioFinalTupla(array):
    mins = []
    for i in range(len(array)-1):
        mins.append((array[i],array[i+1]))
    return mins

def ret_complemento(test, length):
    comp = []
    for i in range(length):
        if i not in test:
            comp.append(i)
    return comp

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
df = pd.read_csv('newLeague2.csv')
#dfIndex = pd.read_csv('newLeague.csv')
#df = normalize(df)
#df.to_csv('newLeague2.csv', index=False)

output=df[['resultBlue','resultRed']]
df.drop('resultBlue', axis=1, inplace=True)
df.drop('resultRed', axis=1, inplace=True)

tamanho = df.shape[0]
dimensao = df.shape[1]
seq_length = 9
comeco_partida = df.loc[df['minute'] == 0]
min_inicios = comeco_partida.index.tolist()
min_inicios_tupla = inicioFinalTupla(min_inicios)
#print('min_inicios_tupla', len(min_inicios_tupla))
test = random.sample(range(len(min_inicios_tupla)), 1254)
test.sort()
#print('test',test)
train = ret_complemento(test, len(min_inicios_tupla))
tempo_test = np.array(min_inicios_tupla)[test]
tempo_train = np.array(min_inicios_tupla)[train]

dataX_Train = []
dataY_Train = []
#print('min_inicios', min_inicios)
for min_ini in tempo_train:
    for i in range(min_ini[0], min_ini[1] - seq_length, 1):
        seq_in = df.iloc[i:i+seq_length,:].values
        seq_out = output.iloc[i+seq_length]
        dataX_Train.append(seq_in)
        dataY_Train.append(seq_out)
        #print(i)
n_patterns = len(dataX_Train)
dataX_Train = np.array(dataX_Train)
X_train= np.reshape(dataX_Train, (n_patterns, seq_length, dataX_Train.shape[2]))
X_train = np.nan_to_num(X_train)
Y_train = np.array(dataY_Train)
Y_train = np.nan_to_num(Y_train)

dataX_Test = []
dataY_Test = []
#print('min_inicios', min_inicios)
for min_ini in tempo_test:
    for i in range(min_ini[0], min_ini[1] - seq_length, 1):
        seq_in = df.iloc[i:i+seq_length,:].values
        seq_out = output.iloc[i+seq_length]
        dataX_Test.append(seq_in)
        dataY_Test.append(seq_out)
        #print(i)
n_patterns = len(dataX_Test)
dataX_Test = np.array(dataX_Test)
X_test = np.reshape(dataX_Test, (n_patterns, seq_length, dataX_Test.shape[2]))
X_test = np.nan_to_num(X_test)
Y_test = np.array(dataY_Test)
Y_test = np.nan_to_num(Y_test)


indTest = getIndexTest(X_test)
X_test = X_test[indTest,:,:]
Y_test = Y_test[indTest]
print('X_test.shape',X_test.shape)
print('Y_test.shape',Y_test.shape)
print('X_train.shape',X_train.shape)
print('Y_train.shape',Y_train.shape)

model = Sequential()
model.add(SimpleRNN(8, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=64, callbacks=callbacks_list, verbose = 1)
