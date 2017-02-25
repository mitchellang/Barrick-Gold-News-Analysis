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
    stems = stem_tokens(tokens, stemmer)
    return stems

def cleanHTML(raw_html):
    cleanR = re.compile('<.*?>')
    cleanText = re.sub(cleanR,'',raw_html)
    return cleanText

def test_prep(text_list):
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
        stop_words = 'english',
        max_features = 85
    )
    tokenized_text = [' '.join(tokenize(w)) for w in text_list]
    corpus_data_features = vectorizer.fit_transform(tokenized_text)
    return corpus_data_features.toarray()

def testing_data():
    url = "http://www.google.ca/finance/company_news?q=TSE:ABX&ei=EOOlWKm6BIe62Aam_4XoCQ&output=rss"
    urlopen = urllib.request.urlopen(url)

    dom = minidom.parse(urlopen)

    xmlTitle = dom.getElementsByTagName('description')

    test_list = []

    for testing in xmlTitle:
        test_list.append((cleanHTML(testing.firstChild.data)))

    prepped_list = test_prep(test_list)
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
        max_features = 85
    )
    combined_list = train_data_df.Text.tolist() + test_data_df.Text.tolist()
    test_list = test_data_df.Text.tolist()
    tokenized_list = [' '.join(tokenize(w)) for w in combined_list]

    #print(combined_list)
    corpus_data_features = vectorizer.fit_transform(tokenized_list)

    corpus_data_features_nd = corpus_data_features.toarray()


    X_train, X_test, y_train, y_test = train_test_split(
        corpus_data_features_nd[0:len(train_data_df)],
        train_data_df.Sentiment,
        train_size=0.85,
        random_state=1234)
    print(X_train)
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    #Uncomment from here
    # log_model = LogisticRegression()
    # log_model = log_model.fit(X=corpus_data_features_nd[0:len(train_data_df)], y=train_data_df.Sentiment)
    # test, original = testing_data()
    # test_pred = log_model.predict(test)
    # original = [' '.join(tokenize(w)) for w in original]
    # raw_data = {'text': original, 'sentiment': test_pred}
    # df = pd.DataFrame(raw_data,columns=['text','sentiment'])
    #
    #
    # print (test_pred)
    # print(df)
    #
    # df.to_csv("output.csv", index=False)
    #To here

    # spl = random.sample(range(len(test_pred)), 15)
    #
    # for text, sentiment in zip(test_data_df.Text[spl], test_pred[spl]):
    #     print(sentiment, text)

main()
