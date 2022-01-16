#%%

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import nltk

# nltk.download()
# nltk.set_proxy('SYSTEM PROXY')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')

# %matplotlib inline
from nltk import tokenize

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from keras.engine.topology import Layer
# from keras import initializers as initializers, regularizers, constraints
# from keras.utils.np_utils import to_categorical
# from keras import optimizers
# from keras.models import Model

#%%

data=pd.read_csv('D:/cis_mbti/mbti_1.csv')
print(data.head())
print(data.shape)

#%%

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")

def cleaning_data(data, remove_stop_words=True):
    list_posts = []
    i=0
    for row in data.iterrows():
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts) #remove urls
        temp = re.sub("[^a-zA-Z.\']", " ", temp) #remove all punctuations except fullstops.
        temp = re.sub(' +', ' ', temp).lower()
        temp=re.sub(r'\.+', ".", temp) #remove multiple fullstops.
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])
        list_posts.append(temp)

    text = np.array(list_posts)
    return text

#%%

clean_text = cleaning_data(data, remove_stop_words=True)
data['clean_text']=clean_text
data = data[['clean_text', 'type']]
print(data.head())
print(data['clean_text'])

#%%

types=data['type']
text=data['clean_text']
tps=data.groupby('type')
print("total types:",tps.ngroups)
print(tps.size())

#%%

max_len=200   # maximum words in a sentence
VAL_SPLIT = 0.2

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
max_features = len(tokenizer.word_index) + 1 # maximum number of unique words
#print(tokenizer.word_index)
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(types)
print("types:",tokenizer2.word_index)


typescode = []
for line in (data['type']):
    #print("line:",line)
    token_list = tokenizer2.texts_to_sequences([line])[0]
    #print("tl:",token_list)
    typescode.append(token_list)
    #print("sequence:", n_gram_sequence)

max_seq_length2 = max([len(x) for x in typescode])
typescode = np.array(pad_sequences(typescode, maxlen = max_seq_length2, padding='pre'))
print(typescode)

input_sequences = []
for line in (data['clean_text']):
    #print("line:",line)
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print("tl:",token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
    #print("sequence:", n_gram_sequence)
print(input_sequences)
#%%

max_seq_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_seq_length, padding='pre'))
print(input_sequences)

xs, labels = input_sequences,typescode
ys = tf.keras.utils.to_categorical(labels, num_classes=max_features, dtype='float64')
X=xs.shape[0]
x_val = xs * VAL_SPLIT
y_val = ys * VAL_SPLIT
print(xs.shape)
print(xs)
print(labels)
print(ys.shape)
print(ys)
print(x_val.shape)
print(y_val.shape)


#test_data=xs[]

#%%

model = Sequential()
model.add(Embedding(max_features, 64, input_length = max_seq_length - 1))
model.add(tf.keras.layers.Conv1D(32, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv1D(32, 1, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv1D(64, 1, padding='same', activation='relu'))
model.add(tf.keras.layers.Conv1D(64, 1, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())

#model.add(Bidirectional(LSTM(64)))
model.add(Dense(max_features, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
history = model.fit(xs, ys, epochs = 500, validation_data=(x_val, y_val), verbose = 1)

#测试集
#test_loss,test_acc=model.evaluate(test_data,test_labels,loss = 'categorical_crossentropy', optimizer ='adam')
#%%

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#%%

model.save('mbti_cov1D.h5')