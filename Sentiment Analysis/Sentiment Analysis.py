from xml.dom import minidom
import urllib.request
import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def cleanHTML(raw_html):
    cleanR = re.compile('<.*?>')
    cleanText = re.sub(cleanR,'',raw_html)
    return cleanText


def processContent(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence.lower())
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return filtered_words

def AFINN():
    sentiment_dictionary = {}
    for line in open('AFINN-111.txt'):
        word, score = line.split('\t')
        sentiment_dictionary[word] = int(score)
    return sentiment_dictionary

def main():
    url = "http://www.google.ca/finance/company_news?q=TSE:ABX&ei=EOOlWKm6BIe62Aam_4XoCQ&output=rss"
    urlopen = urllib.request.urlopen(url)

    dom = minidom.parse(urlopen)

    xmlTitle = dom.getElementsByTagName('description')
    #print(xmlTitle.firstChild.data)

    sentiment_dictionary = AFINN()

    for testing in xmlTitle:
        sentenceTokenized = sent_tokenize(cleanHTML(testing.firstChild.data))
        totalScore = 0

        for sentence in sentenceTokenized:

            stopWordsRemoved = processContent(sentence)
            sentimentScore = sum(sentiment_dictionary.get(word, 0) for word in stopWordsRemoved)
            totalScore += sentimentScore
            print(sentence)

        print(totalScore)
        if totalScore < 0:
            print('bad')
        elif totalScore >0:
            print('positive')
        else:
            print('neutral')

main()
