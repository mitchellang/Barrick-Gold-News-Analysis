from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from xml.dom import minidom
import urllib.request
import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer

test_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

test_data_file_name = 'test_data.csv'
train_data_file_name = 'train_data.csv'

test_data_f = urllib.request.urlretrieve(test_data_url, test_data_file_name)
train_data_f = urllib.request.urlretrieve(train_data_url, train_data_file_name)

test_data_df = pd.read_csv(test_data_file_name, header=None, delimiter="\t", quoting=3)
test_data_df.columns = ["Text"]
train_data_df = pd.read_csv(train_data_file_name, header=None, delimiter="\t", quoting=3)
train_data_df.columns = ["Sentiment", "Text"]

X_train, X_test, y_train, y_test = train_test_split(
    train_data_df.Text,
    train_data_df.Sentiment,
    train_size=0.85,
    random_state=1234)

print(X_train)