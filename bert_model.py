from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import regex as re
import transformers
from tensorflow.keras import backend as K
import plotly.express as px

data = pd.read_csv('D:\cis_mbti\mbti_1.csv')
data.head()

# %%

# Check if TPU is available
use_tpu = True
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.MirroredStrategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# %%

px.pie(data, names='type', title='Distribution of personality types', hole=0.3)

# %%

# %%

data['type'].value_counts()

# %%

def clean_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(data.posts):
        # removing links from text data
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)

        # removing other symbols
        sentence = re.sub('[^0-9a-z]', ' ', sentence)

        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text


# %%

data.posts = clean_text(data)
data

# %% md

# %%

# Split dataset
from sklearn.model_selection import train_test_split

posts = data['posts'].values
labels = data['type'].values
train_data, test_data = train_test_split(data, random_state=0, test_size=0.2)

train_size = len(train_data)
test_size = len(test_data)
train_size, test_size

# %%

# Initialize Bert tokenizer and masks
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

bert_model_name = 'bert-base-uncased'

tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
MAX_LEN = 1800


def tokenize_sentences(sentences, tokenizer, max_seq_len=1800):
    tokenized_sentences = []

    for sentence in tqdm(sentences):
        tokenized_sentence = tokenizer.encode(
            sentence,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_seq_len,  # Truncate all sentences.
        )

        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences


def create_attention_masks(tokenized_and_padded_sentences):
    attention_masks = []

    for sentence in tokenized_and_padded_sentences:
        att_mask = [int(token_id > 0) for token_id in sentence]
        attention_masks.append(att_mask)

    return np.asarray(attention_masks)


train_input_ids = tokenize_sentences(train_data['posts'], tokenizer, MAX_LEN)
train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                padding="post")
train_attention_masks = create_attention_masks(train_input_ids)

test_input_ids = tokenize_sentences(test_data['posts'], tokenizer, MAX_LEN)
test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
test_attention_masks = create_attention_masks(test_input_ids)

# %%

# train_masks,test_masks, _, _ = train_test_split(attention_masks, labels, random_state=0, test_size=0.2)

# %%

# Create train and test datasets
BATCH_SIZE = 32
NR_EPOCHS = 20
# def create_dataset(data_tuple, epochs=1, batch_size=32, buffer_size=10000, train=True):
#    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
#    if train:
#        dataset = dataset.shuffle(buffer_size=buffer_size)
#    dataset = dataset.repeat(epochs)
#    dataset = dataset.batch(batch_size)
#    if train:
#        dataset = dataset.prefetch(1)

#   return dataset

# train_dataset = create_dataset((train_inputs, train_masks, train_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE)
# test_dataset = create_dataset((test_inputs, test_masks, test_labels), epochs=NR_EPOCHS, batch_size=BATCH_SIZE, train=False)



# %%


# from transformers import TFBertModel

# from tensorflow.keras.layers import Dense, Flatten

# class BertClassifier(tf.keras.Model):
#        def __init__(self, bert: TFBertModel, num_classes: int):
#            super().__init__()
#            self.bert = bert
#            self.classifier = Dense(16, activation='softmax')

#        @tf.function
#        def call(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
#            outputs = self.bert(input_ids,
#                                   attention_mask=attention_mask,
#                                   token_type_ids=token_type_ids,
#                                   position_ids=position_ids,
#                                   head_mask=head_mask)
#            cls_output = outputs[1]
#            cls_output = self.classifier(cls_output)

#            return cls_output


# with strategy.scope():
#    model = BertClassifier(TFBertModel.from_pretrained(bert_model_name), len(label_cols))

# %%

# Define f1 functions for evaluation


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# %%

def create_model():
    input_word_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32,
                                           name="input_word_ids")
    bert_layer = transformers.TFBertModel.from_pretrained('bert-large-uncased')
    bert_outputs = bert_layer(input_word_ids)[0]
    pred = tf.keras.layers.Dense(16, activation='softmax')(bert_outputs[:, 0, :])

    model = tf.keras.models.Model(inputs=input_word_ids, outputs=pred)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00002), metrics=['accuracy', f1_m, precision_m, recall_m])
    return model


# %%

# use_tpu = False
# if use_tpu:
#     # Create distribution strategy
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
#
#     # Create model
#     with strategy.scope():
#         model = create_model()
# else:
#     model = create_model()
#
# model.summary()

# %%
model = create_model()
types = np.unique(data.type.values)


def get_type_index(string):
    return list(types).index(string)


# %%

train_data['type_index'] = data['type'].apply(get_type_index)
train_data

# %%

one_hot_labels = tf.keras.utils.to_categorical(train_data.type_index.values, num_classes=16)

# %%


model.fit(np.array(train_input_ids), one_hot_labels, verbose=1, epochs=NR_EPOCHS, batch_size=BATCH_SIZE,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

# %% md



# %%

test_data['type_index'] = data['type'].apply(get_type_index)
test_data

# %%

test_labels = tf.keras.utils.to_categorical(test_data.type_index.values, num_classes=16)

# %%

model.evaluate(np.array(test_input_ids), test_labels)

# %% md



# %%

cols = data['type'].unique()
cols = cols.tolist()

colnames = ['sentence']
colnames = colnames + cols

# %%


df_predict = pd.DataFrame(columns=colnames)
sentence = "Time to debate on it. Strike at the weakest point and make others cry with facts"

df_predict.loc[0, 'sentence'] = sentence

# %%

sentence_inputs = tokenize_sentences(df_predict['sentence'], tokenizer, MAX_LEN)
sentence_inputs = pad_sequences(sentence_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                padding="post")
prediction = model.predict(np.array(sentence_inputs))
df_predict.loc[0, cols] = prediction

df_predict

# %% md

