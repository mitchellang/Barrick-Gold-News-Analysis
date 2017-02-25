from xml.dom import minidom
import urllib.request
import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sklearn.svm
from sklearn.metrics import confusion_matrix
import sklearn.naive_bayes
import random

stemmer = PorterStemmer()

#stemming tokens
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#cleanup text and tokenizing text
def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    #stems = stem_tokens(tokens, stemmer)
    return tokens

def cleanHTML(raw_html):
    cleanR = re.compile('<.*?>')
    cleanText = re.sub(cleanR,'',raw_html)
    return cleanText

def test_prep(text_list, vectorizer):

    tokenized_text = [' '.join(tokenize(w)) for w in text_list]
    corpus_data_features = vectorizer.transform(tokenized_text)
    return corpus_data_features.toarray()

def testing_data(vectorizer):
    url = "http://www.google.ca/finance/company_news?q=TSE:ABX&ei=EOOlWKm6BIe62Aam_4XoCQ&output=rss"
    urlopen = urllib.request.urlopen(url)

    dom = minidom.parse(urlopen)

    xmlTitle = dom.getElementsByTagName('description')

    test_list = []

    for testing in xmlTitle:
        test_list.append((cleanHTML(testing.firstChild.data)))

    prepped_list = test_prep(test_list, vectorizer)
    return prepped_list, test_list

#print(testing_data())

def main():
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

    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
        stop_words = 'english',
    )

    X_train, X_test, y_train, y_test = train_test_split(
        train_data_df.Text,
        train_data_df.Sentiment,
        train_size=0.85,
        random_state=1234)

    combined_list = X_train.tolist()
    test_list = X_test.tolist()
    result_list = y_train.tolist()
    test_result_list = y_test.tolist()

    tokenized_list = [' '.join(tokenize(w)) for w in combined_list]
    tokenized_test = [' '.join(tokenize(w)) for w in test_list]

    #print(combined_list)
    corpus_data_features = vectorizer.fit_transform(tokenized_list).toarray()
    corpus_test_features = vectorizer.transform(tokenized_test).toarray()

    #corpus_data_features_nd = corpus_data_features.toarray()


    log_model = LogisticRegression()
    log_model = log_model.fit(X=corpus_data_features, y=result_list)
    # y_pred = log_model.predict(corpus_test_features)

    # print(classification_report(test_result_list, y_pred))
    # print(accuracy_score(test_result_list, y_pred))
    # print(confusion_matrix(test_result_list, y_pred, labels = [0,1]))



    #Uncomment from here
    # log_model = LogisticRegression()
    # log_model = log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
    test, original = testing_data(vectorizer)
    test_pred = log_model.predict(test)
    original = [' '.join(tokenize(w)) for w in original]
    raw_data = {'text': original, 'sentiment': test_pred}
    df = pd.DataFrame(raw_data,columns=['text','sentiment'])


    # print (test_pred)
    # print(df)

    df.to_csv("output.csv", index=False)
    #To here

    # spl = random.sample(range(len(test_pred)), 15)
    #
    # for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    #     print(sentiment, text)

main()