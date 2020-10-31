import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle
import tensorflow, json
from tensorflow import keras
from sklearn import preprocessing
from support import get_languages
from sklearn.preprocessing import MinMaxScaler
import random

master_data = np.load('master_data.npy',allow_pickle=True)
lang = get_languages()
lang_short = ['en','fr','it','es']

shuffle(master_data)
shuffle(master_data)
X_master = np.array(master_data[:,0], dtype=np.float32)
y_master = np.array(master_data[:,1], dtype=np.float32)

print(type(X_master))
print(type(y_master))
print(X_master[0])
print(y_master[1])
np.save('X_train.npy',X_master)
np.save('y_train.npy',y_master)
print(np.shape(X_master))
print(np.shape(y_master))


