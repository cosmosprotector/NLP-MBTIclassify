# Data Analysis
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import pickle as pkl
from scipy import sparse

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# import wordcloud
# from wordcloud import WordCloud, STOPWORDS

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

# loading dataset
data_set = pd.read_csv("D:/cis_mbti/mbti_1.csv")
data_set.tail()



def preprocess_text(df, remove_special=True):
    texts = df['posts'].copy()
    labels = df['type'].copy()

    # Remove links
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|", " ") + " "))

    # Keep the End Of Sentence characters
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\.', ' EOSTokenDot ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'\?', ' EOSTokenQuest ', x + " "))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'!', ' EOSTokenExs ', x + " "))

    # Strip Punctation
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[\.+]', ".", x))

    # Remove multiple fullstops
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove Non-words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Convert posts to lowercase
    df["posts"] = df["posts"].apply(lambda x: x.lower())

    # Remove multiple letter repeating words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'([a-z])\1{2,}[\s|\w]*', '', x))

    # Remove very short or long words
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{0,3})?\b', '', x))
    df["posts"] = df["posts"].apply(lambda x: re.sub(r'(\b\w{30,1000})?\b', '', x))

    # Remove MBTI Personality Words - crutial in order to get valid model accuracy estimation for unseen data.
    if remove_special:
        pers_types = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ',
                      'ESTP', 'ESFP', 'ESFJ', 'ESTJ']
        pers_types = [p.lower() for p in pers_types]
        p = re.compile("(" + "|".join(pers_types) + ")")

    return df



# Preprocessing of entered Text
new_df = preprocess_text(data_set)

# Remove posts with less than X words
min_words = 15
print("Before : Number of posts", len(new_df))
new_df["no. of. words"] = new_df["posts"].apply(lambda x: len(re.findall(r'\w+', x)))
new_df = new_df[new_df["no. of. words"] >= min_words]

print("After : Number of posts", len(new_df))

print(new_df.head())


## Splitting into X and Y feature

# Converting MBTI personality (or target or Y feature) into numerical form using Label Encoding
# encoding personality type
enc = LabelEncoder()
new_df['type of encoding'] = enc.fit_transform(new_df['type'])

target = new_df['type of encoding']
print(target)
print(type(target))
target = np.array(target)
print(target)
print(target.shape)
print(type(target))

print(new_df.head(15))

# The python natural language toolkit library provides a list of english stop words.
print(stopwords.words('english'))

print(new_df["posts"].head(15))

# Vectorizing the posts for the model and filtering Stop-words
vect = TfidfVectorizer(stop_words='english')

# Converting posts (or training or X feature) into numerical form by count vectorization
train = vect.fit_transform(new_df["posts"])

print(train)



print(train.shape)

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.4, stratify=target, random_state=42)
print((X_train.shape), (y_train.shape), (X_test.shape), (y_test.shape))
print(y_test.dtype)
print(y_test)




#load trained_models
import joblib

RF = joblib.load("RF_model.m")
LR = joblib.load("lr.pkl")

import pickle

# load
xgb = pickle.load(open("xgb.pkl", "rb"))


# test
def vote(X_test,y_test):
    vote_res=[]
    print(len(vote_res))
    count=0
    for i in range(0,X_test.shape[0]):
        xgb_res=xgb.predict(X_test[i])
        print("xgb:",xgb_res)
        LR_res = LR.predict(X_test[i])
        print("LR:", LR_res)
        RF_res = RF.predict(X_test[i])
        print("RF:", RF_res)
        if xgb_res==LR_res:
            vote_res.append(xgb_res)
        elif xgb_res==RF_res:
            vote_res.append(xgb_res)
        elif RF_res==LR_res:
            vote_res.append(LR_res)
        else:
            vote_res.append(xgb_res)
        print(vote_res[i])
        print(y_test[i])
    for i in range(0, X_test.shape[0]):
        if vote_res[i]==y_test[i]:
            count=count+1
    acc = count/y_test.shape[0]
    return acc
#xgb.predict(test)


#投票
vote_acc=vote(X_test,y_test)
print(vote_acc)
