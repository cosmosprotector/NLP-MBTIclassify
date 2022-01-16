# Data Analysis
import pandas as pd
import numpy as np
import sklearn.metrics as sm

from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Text Processing
import re
import itertools
import string
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Machine Learning packages
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.cluster as cluster
from sklearn.manifold import TSNE

# Model training and evaluation
from sklearn.model_selection import train_test_split

#Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

#Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import classification_report

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

#loading dataset
data = pd.read_csv("D:/cis_mbti/mbti_1.csv")
data.tail()


def get_types(row):
    t = row['type']

    I = 0;
    N = 0
    T = 0;
    J = 0

    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    else:
        print('I-E not found')

    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0
    else:
        print('N-S not found')

    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    else:
        print('T-F not found')

    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0
    else:
        print('J-P not found')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


data = data.join(data.apply(lambda row: get_types(row), axis=1))
print(data.head(5))

print ("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
print ("Intuition (N) / Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print ("Thinking (T) / Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print ("Judging (J) / Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

lemmatiser = WordNetLemmatizer()

# Remove the stop words for speed
useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
       'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Or we can use Label Encoding (as above) of this unique personality type indicator list
# from sklearn.preprocessing import LabelEncoder
# unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
#        'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
# lab_encoder = LabelEncoder().fit(unique_type_list)

# Splitting the MBTI personality into 4 letters and binarizing it

b_Pers = {'I':0, 'E':1, 'N':0, 'S':1, 'T':0, 'F':1, 'J':0, 'P':1}
b_Pers_list = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {0:'T', 1:'F'}, {0:'J', 1:'P'}]

def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]

#To show result output for personality prediction
def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s

list_personality_bin = np.array([translate_personality(p) for p in data.type])
print("Binarize MBTI list: \n%s" % list_personality_bin)


def pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        # check code working
        # i+=1
        # if (i % 500 == 0 or i == 1 or i == len_data):
        #     print("%s of %s rows" % (i, len_data))

        # Remove and clean comments
        posts = row[1].posts

        # Remove url links
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        # Remove multiple letter repeating words
        temp = re.sub(r'([a-z])\1{2,}[\s|\w]*', '', temp)

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        # transform mbti to binary vector
        type_labelized = translate_personality(row[1].type)  # or use lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        # the cleaned data temp is passed here
        list_posts.append(temp)

    # returns the result
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


list_posts, list_personality = pre_process_text(data, remove_stop_words=True, remove_mbti_profiles=True)

print("Example :")
print("\nPost before preprocessing:\n\n", data.posts[0])
print("\nPost after preprocessing:\n\n", list_posts[0])
print("\nMBTI before preprocessing:\n\n", data.type[0])
print("\nMBTI after preprocessing:\n\n", list_personality[0])



# Vectorizing the posts for the model and filtering Stop-words
vect = TfidfVectorizer(stop_words='english')

# Converting posts (or training or X feature) into numerical form by count vectorization
X = vect.fit_transform(list_posts)


print(X.shape)

personality_type = [ "First label(I/E)   ", "Second label(N/S)   ",
                   "Third label(T/F)   ", "Fourth label(J/P)   "  ]



# Random Forest model for MBTI dataset
# Individually training each mbti personlity type
print("\nrandom forest")
for l in range(len(personality_type)):
    Y = list_personality[:, l]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    warnings.filterwarnings("ignore")
    # fit model on training data
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)

    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
    # # 混淆矩阵 矩阵  行：分类 列：分类
    # cm = sm.confusion_matrix(y_test, predictions)
    # print("---------------混淆矩阵\n", cm)

    # cp = sm.classification_report(y_test, predictions)
    # print("---------------分类报告\n", cp)
    # acc = np.sum(y_test==predictions) / y_test.size
    # print(acc)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [("%s Confusion matrix, without normalization"%(personality_type[l]), None),
                      ("%s Normalized confusion matrix"%(personality_type[l]), 'true')]
    for title, normalize in titles_options:
        disp = sm.plot_confusion_matrix(model, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()


# xgboost model for MBTI dataset
# Individually training each mbti personlity type
print("\nxgboost")
for l in range(len(personality_type)):
    Y = list_personality[:, l]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    warnings.filterwarnings("ignore")
    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)

    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))
    np.set_printoptions(precision=2)
    titles_options = [("%s Confusion matrix, without normalization" % (personality_type[l]), None),
                      ("%s Normalized confusion matrix" % (personality_type[l]), 'true')]
    for title, normalize in titles_options:
        disp = sm.plot_confusion_matrix(model, X_test, y_test,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

# Linear regression model for MBTI dataset
# Individually training each mbti personlity type
print("\nLinear regression")
for l in range(len(personality_type)):
    Y = list_personality[:, l]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

    warnings.filterwarnings("ignore")
    # fit model on training data
    model = LogisticRegression(solver='liblinear', class_weight="balanced")
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)

    print("%s Accuracy: %.2f%%" % (personality_type[l], accuracy * 100.0))

    np.set_printoptions(precision=2)
    titles_options = [("%s Confusion matrix, without normalization" % (personality_type[l]), None),
                      ("%s Normalized confusion matrix" % (personality_type[l]), 'true')]
    for title, normalize in titles_options:
        disp = sm.plot_confusion_matrix(model, X_test, y_test,
                                        cmap=plt.cm.Blues,
                                        normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()