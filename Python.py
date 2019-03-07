# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:29:58 2019

@author: mounika
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#changing the directory
os.chdir("C:\\Users\\mounika\\Desktop")
#importing the text file
df1=open('tweets.txt','r',encoding='utf8')
df1=df1.readlines()

#Separating the hasttag words into proper words++++z
Content=open('Wordlist.txt','r')
Words=Content.readlines()
Wordlist=[x.rstrip('\n') for x in Words]

def hashwords(sent,Wordlist):
    output=[]
    terms=sent.split(' ')
    for term in terms:
        if term[0]=='#':#if the word starts with #
            output.append(parse(term,Wordlist))#arsing the element
        else:
            output.append((term))#else appending it
    return(output)

def parse(term,Wordlist):
    words=[]
    tags=term[1:].split("-")
    for tag in tags:
        word=find_word(tag,Wordlist)    
        while word!=None and len(tag)>0:
            words.append(word)
            if len(tag)==len(word):
                break
            tag=tag[len(word):]
            word=find_word(tag,Wordlist)    
    return(" ".join(words))
        
        
def find_word(token, Wordlist):
    i = len(token) + 1
    while i > 1:
        i=i-1
        if token[:i] in Wordlist:
            return token[:i]
    return None


df1=pd.DataFrame(df1,columns=['Comments'])
df1[0:10]
df3=df1.iloc[0:10,:]

import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
 
#text cleaning
from nltk import word_tokenize
def tokenize(text):
    tokens=hashwords(text.lower(),Wordlist)
    review=[(x) for x in tokens]
    corpus=[]
    for x in review:
        if re.search('[A-Za-z]',x):
            corpus.append(x)
    return corpus
from nltk.corpus import stopwords
stoplist = set(stopwords.words("english"))

#Termfrequency inverse documnetary frequency
from sklearn.feature_extraction.text import TfidfVectorizer 
a=TfidfVectorizer(min_df=0.005,max_df=0.95,ngram_range=(2,3),max_features=2000,
                  use_idf=True,tokenizer=tokenize,stop_words=stoplist)
v1=a.fit_transform(df3['Comments'])
V2=a.get_feature_names()
#applying latent dirhlet allocation

from sklearn.decomposition import LatentDirichletAllocation
LDA=LatentDirichletAllocation(n_topics=3,random_state=0,learning_method='online')
l1=LDA.fit_transform(v1)




 
        
        
        
        
        
        
        
        
        
