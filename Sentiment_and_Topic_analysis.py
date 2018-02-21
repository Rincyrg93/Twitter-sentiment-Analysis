#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:12:10 2017

@author: rincygeorge
"""


import json
import nltk
import preprocessor as p
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from vocabulary.vocabulary import Vocabulary as vb
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import csv

states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

#function for pre-processing
def parse_text(txt):
    txt = txt.lower()  #converting to lowercase
    #removing punctuation and digits
    p=string.punctuation    
    d=string.digits
    tables=str.maketrans(p, len(p) *" ")
    text1=txt.translate(tables)
    tables=str.maketrans(d, len(d) *" ")
    text1=text1.translate(tables)
    
    words=word_tokenize(text1) #tokenization
    #lemmatization
    wordnet_lemmatizer = WordNetLemmatizer() 
    words1=[wordnet_lemmatizer.lemmatize(token) for token in words]
    
    #removing stopwords
    stopwords=nltk.corpus.stopwords.words("English")
    extra_stopwords=['rt','RT','TakeTheKnee','taketheknee','TakeAKnee','takeaknee'] #adding stopwords
    stopwords.extend(extra_stopwords)
    words=[w for w in words1 if w not in stopwords]
    return " ".join(words)


corpus= {}
corpus_text=[]
# Getting tweets location-wise
with open("/Users/rincygeorge/Desktop/tweet_data_20000.json") as infile:
    tweets_list=json.load(infile)
    for tweets in tweets_list:
        single_tweet=json.loads(tweets)
        if 'place' in single_tweet:
            if single_tweet['place']:
                if 'full_name' in single_tweet['place']:
                    if ',' in single_tweet['place']['full_name']:
                        x=single_tweet['place']['full_name'].split(',')[1]
                        state_tweet=x.strip()
                        if state_tweet in states:
                            f=parse_text(single_tweet['text'])
                            
                            if state_tweet in corpus:
                                corpus[state_tweet].append(f)
                                
                            else:
                                corpus[state_tweet]=[f]
                                
#function for sentiment score
def SentimentScore(txt):
    sentiment_parameter=TextBlob(txt).sentiment.polarity
    return sentiment_parameter

#Calculating the sentiment score using textBlob
sentiment_score_dict={}   
for key in corpus:
    score=0
    for item1 in corpus[key]:
        score=score+ SentimentScore(item1)
    sentiment_score_dict[key]=score

#to write the State-wise sentiment score into a csv file
with open("/Users/rincygeorge/Desktop/DS/twitter_prog/tweet_final_sheet.csv", 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in sentiment_score_dict.items():
       writer.writerow([key, value])

#NMF- Topic Modelling       
with open("/Users/rincygeorge/Desktop/tweet_data_20000.json") as infile:
    tweets_list=json.load(infile)
    for tweets in tweets_list:
        single_tweet=json.loads(tweets)
        f=parse_text(single_tweet['text'])
        corpus_text.append(f)
             
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm = vectorizer.fit_transform(corpus_text)

num_topics = 10
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
tfidf_feature_names = vectorizer.get_feature_names()
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([tfidf_feature_names[i] for i in word_idx])
for t in range(len(topic_words)):
    with open("/Users/rincygeorge/Desktop/topics.txt",'a') as writefile:
        writefile.write("Topic {}: {}".format(t, ' '.join(topic_words[t][:15]))) 
        writefile.write('\n')
        
    

