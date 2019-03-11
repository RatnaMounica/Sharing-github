# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:54:52 2019

@author: mounika
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#Separating the hasttag words into proper words++++z
Content=open('Wordlist.txt','r')
Words=Content.readlines()
Wordlist=[x.rstrip('\n') for x in Words]

"""Whenthe word starts with # trying to split into proper words using 
parse userdefined function or just appending back to output"""
def hashwords(sent,Wordlist):
    output=[]
    terms=sent.split()
    for term in terms:
        #print(term)
        if term[0]=='#':
            output.append(parse(term,Wordlist))      
        else:
            output.append((term))
    return(output)
    
"""Parsing the word by removing the hashtag and finding the words with wordlist"""
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

#changing the directory
os.chdir("C:\\Users\\mounika\\Desktop")
#importing the text file
df1=open('tweets.txt','r',encoding='utf8')
df1=df1.readlines()
df1=pd.DataFrame(df1,columns=['Comments'])
#removing Emcons
import re
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
a=[]
for i in range(0,100041):
    a.append(emoji_pattern.sub(r'', df1['Comments'][i])) 
df2=pd.DataFrame(a,columns=['Comments'])
del a
#Validation
print("Before: "+df1.iloc[7,:][0])   
print("After:  "+df2.iloc[7,:][0]) 
del df1



#Coverting other languages into english
"""from textblob import TextBlob
L=[]
for i in range(0,255):
    l=df2['Comments'][i]
    blob=TextBlob(l)
    p=blob.detect_language()
    if(p=='en'):
        L.append(l)
    else:
        a=blob.translate(to="es")
        x=str(a)
        L.append(x)
df3=pd.DataFrame(L,columns=['Comments'])
#Validation
print("Before: "+df2.iloc[10,:][0])   
print("After:  "+df3.iloc[10,:][0]) """


#removing https 
L=[]
L= df2['Comments'].apply(lambda x: " ".join(x for x in x.split() if x not in (x[x.find("http://"):].rstrip())))
df4=pd.DataFrame(L,columns=['Comments'])
#Validation
print("Before: "+df2.iloc[3,:][0])   
print("After:  "+df4.iloc[3,:][0])
del df2

#splitting the Hashtag words into proper words
L=[]
df5=[]
df5=df4['Comments'].str.lower()
df5=pd.DataFrame(df5,columns=['Comments'])
for i in range(0,100041):
      L.append([hashwords(df5['Comments'][i],Wordlist)])
df6=pd.DataFrame(L,columns=['Comments'])
#validations
print("Before: "+df5.iloc[7,:][0])   
print(df6.iloc[7,:][0])
del df5


#Removing Punctatuation
df7=[]
import re,string
df7= df6['Comments'].apply(lambda x:''.join([i for i in x
                                                  if i not in string.punctuation]))
#validations
print("Before: "+df6.iloc[20,:][0])   
print(df7.iloc[20,:][0])



#Textcleaning-stopwords
l=[]
df8=[]
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
extended_words=['american','seen','sniper','le','see','wanna','watching','movie','american sniper','americansniper','sniper!','"american','sniper"',"'american","sniper'","sniper':",]
stop=stopwords+extended_words
l = df8['Comments'].apply(lambda x: " ".join(x for x in x if x not in stop))
df8=pd.DataFrame(l,columns=['Comments'])
#validations
print(df8.iloc[3,:][0])  
print("After:  "+df8.iloc[3,:][0])
print(df8.iloc[29,:][0])  
print("After:  "+df8.iloc[29,:][0])



#text cleaning --Reoving numbers and special characters
df10=[]
df9=[]
df10= df8['Comments'].str.replace('[^A-Za-z]',' ')
df9=pd.DataFrame(df10,columns=['Comments'])
#validations
print(df8.iloc[13,:][0])  
print("After:  "+df9.iloc[13,:][0])



#srip white spaces
df11=[]
a=[]
a=df9['Comments'].apply(lambda x:" ".join(x for x in x.split()))
df11=pd.DataFrame(a,columns=['Comments'])
#validations
print(df9.iloc[13,:][0])  
print("After:  "+df11.iloc[13,:][0])


#lemitization
df12=[]
t=[]
from textblob import Word
t= df11['Comments'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df12=pd.DataFrame(t,columns=['Comments'])


#Wordcloud
all_words = ' '.join([text for text in df12['Comments']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#By visulaixzing the results the three main personalities 
#Bradley Cooper
#clint eastwood
#chris kyle
def word_in_text(word,text):
        if word in text:
            return text
        else:
            pass
#Tweets for Bradley Cooper    
lbradley=[]
word='bradley'
for i in range(0,100041):
    str=df11.iloc[i,:][0]
    k=word_in_text(word, str)
    if k!=None:
        lbradley.append(k)
dfb=[]
dfb=pd.DataFrame(lbradley,columns=['bradley'])
print("Total tweets :" +"  ")
print( dfb.shape[0])
#sentiment anlaysi
from textblob import TextBlob
dfb['sentiment'] = dfb['bradley'].apply(lambda x: TextBlob(x).sentiment[0] )
print("Postive comments")
print(dfb[dfb['sentiment']>0].shape[0])

#Tweets for clint eastwood
lclint=[]
word='clint'
for i in range(0,100041):
    str=df11.iloc[i,:][0]
    k=word_in_text(word, str)
    if k!=None:
        lclint.append(k)
dfc=[]
dfc=pd.DataFrame(lclint,columns=['clint'])
print("Total tweets :" +"  ")
print( dfc.shape[0])
#sentiment anlaysi
dfc['sentiment'] = dfc['clint'].apply(lambda x: TextBlob(x).sentiment[0] )
print("Postive comments")
print(dfc[dfc['sentiment']>0].shape[0])


#Tweets for chris kyle
lchris=[]
word='chris'
for i in range(0,100041):
    str=df11.iloc[i,:][0]
    k=word_in_text(word, str)
    if k!=None:
        lchris.append(k)

dfch=[]
dfch=pd.DataFrame(lchris,columns=['chris'])
print("Total tweets :" +"  ")
print( dfch.shape[0])
#sentiment anlaysi
dfch['sentiment'] = dfch['chris'].apply(lambda x: TextBlob(x).sentiment[0] )
print("Postive comments")
print(dfch[dfch['sentiment']>0].shape[0])
dfchPostive=pd.DataFrame()
dfchPostive=dfch[dfch['sentiment']>0]
