import re
import pandas as pd
import numpy as np
filename = '_cleaned.txt'
langnames = ['en','fr','it','es','de']
langs = ['English','French', 'Italian','Spanish', 'German']
#%%
for i in range(len(langnames)):
    file = str(langnames[i] +str(filename))
    f = open(file,'r',encoding='utf-8').read()
    f = re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '',f.lower())
    sent = []
    offset = 1
    for j in range(0,100000,100):
        sent.append(f[j*offset: j*offset +100])
    langnames[i] = pd.DataFrame(sent)
    langnames[i]['language'] = langs[i]
    langnames[i].columns = ['sentence','language']
#%%
x = langnames[0]
#%%
# append all dataframes to x
for i in range(1,len(langnames)):
    x = x.append(langnames[i],ignore_index=True)

#%%
print(x.shape)
#%%
from sklearn.model_selection import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(test_size = 0.2, random_state=42)
#%%
X = x['sentence']
Y = x['language']

#%%
allwords = (' '.join([sentence for sentence in X])).split()
#%%
allwords = list(set(allwords))
#%%
for i,j in ss.split(X,Y):
    X_train,X_test = X[i],X[j]
    Y_train, Y_test = Y[i],Y[j]
#%%
languages = set(Y)
#%%

def create_lookup_tables(text):
    
    word_num= {word: i for i, word in enumerate(text)}
    num_word = {value:key for key, value in word_num.items()}
    
    return word_num, num_word
#%%
allwords.append('<UNK>')
#%%

vocabint , intvocab = create_lookup_tables(allwords)
langint, intlang = create_lookup_tables(languages)

#%%
def convert_to_int(data, data_int):
    """Converts all our text to integers
    :param data: The text to be converted
    :return: All sentences in ints
    """
    all_items = []
    for sentence in data: 
        all_items.append([data_int[word] if word in data_int else data_int["<UNK>"] for word in sentence.split()])
    
    return all_items
#%%
    
X_test_encoded = convert_to_int(X_test, vocabint)
X_train_encoded = convert_to_int(X_train, vocabint)
#%%
Y_data = convert_to_int(Y_test, langint)
#%%
from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder()
ohc.fit(Y_data)
#%%
Y_train_encoded = ohc.fit_transform(convert_to_int(Y_train, langint)).toarray()
Y_test_encoded = ohc.fit_transform(convert_to_int(Y_test, langint)).toarray()

#%%
print(Y_train_encoded[:10],'\n',Y_train[:10])
#%%
import warnings
import tensorflow as tf
#%%
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
#%%
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence    
#%%    
dropout_factor = 0.5

#%%
X_train_pad = sequence.pad_sequences(X_train_encoded, maxlen=200)
X_test_pad = sequence.pad_sequences(X_test_encoded, maxlen=200)
#%%
model = Sequential()
    
model.add(Embedding(len(vocabint), 300, input_length=200))
model.add(LSTM(256, return_sequences=True, recurrent_dropout=dropout_factor))
model.add(Dropout(0.5))
model.add(LSTM(256, recurrent_dropout=dropout_factor))
model.add(Dropout(0.5))
model.add(Dense(len(languages), activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#%%    
    
model.fit(X_train_pad, Y_train_encoded, epochs=3, batch_size=256)
#%%
def process_sentence(sentence):
    '''Removes all special characters from sentence. It will also strip out
    extra whitespace and makes the string lowercase.
    '''
    return re.sub(r'[\\\\/:*«`\'?¿";!<>,.|]', '', sentence.lower().strip())
#%%

def predict_sentence(sentence):
    """Converts the text and sends it to the model for classification
    :param sentence: The text to predict
    :return: string - The language of the sentence
    """
    
    # Clean the sentence
    sentence = process_sentence(sentence)
    
    # Transform and pad it before using the model to predict
    x = np.array(convert_to_int([sentence], vocabint))
    x = sequence.pad_sequences(x, maxlen=200)
    
    prediction = model.predict(x)
    
    # Get the highest prediction
    lang_index = np.argmax(prediction)
    
    return intlang[lang_index]

#%%
predict_sentence('Wir sind eine ganz normale Familie. Ich wohne zusammen mit meinen Eltern, meiner kleinen Schwester Lisa und unserer Katze Mick. Meine Großeltern wohnen im gleichen Dorf wie wir. Oma Francis arbeitet noch. ')
#%%

model.save('C:\\Users\\sahas\\Desktop\\newlang.h5')
#%%