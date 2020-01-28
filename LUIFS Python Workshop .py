#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries needed

import os
import csv
import time
import random
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import parser
import timeit
import pickle
import datetime as dt
import wrds
import nltk
from nltk.tokenize import RegexpTokenizer, sent_tokenize
nltk.download('punkt')
import matplotlib.pyplot as plt
import pandas_profiling
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from dateutil.relativedelta import *
from pandas.tseries.offsets import *
from scipy import stats


# In[2]:


os.getcwd()


# In[3]:


os.chdir('/Users/user/Documents/Uni 19:20/AcF351/Coursework')


# #### Import LM sentiment dictionary (created for financial statements)

# In[4]:


#Import the negative and positive dictionaries https://drive.google.com/file/d/15UPaF2xJLSVz8DYuphierz67trCxFLcl/view
##### Words from Loughran and McDonald
# negative 
neg_dict_LM = ""
neg_dict_LM = pd.read_csv(r'lm_negative.csv',encoding = 'ISO-8859-1', names=['lm_negative'])['lm_negative'].values.tolist()
neg_dict_LM = str(neg_dict_LM)
neg_dict_LM = neg_dict_LM.lower()

# positive
pos_dict_LM = ""
pos_dict_LM = pd.read_csv(r'lm_positive.csv', encoding = 'ISO-8859-1', names=['lm_positive'])['lm_positive'].values.tolist()
pos_dict_LM = str(pos_dict_LM)
pos_dict_LM = pos_dict_LM.lower()


# #### Definitions for textual analysis

# In[5]:


#StopWords from Loughran and McDonald - https://sraf.nd.edu/textual-analysis/resources/

stopWordsFile = r'StopWords_Generic.txt'
#Loading stop words dictionary for removing stop words
with open(stopWordsFile ,'r') as stop_words:
    stopWords = stop_words.read().lower()
stopWordList = stopWords.split('\n')
stopWordList[-1:] = []


# In[6]:


stopWordList


# In[7]:


# Tokenizeing module and filtering tokens using stop words list, removing punctuations
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words


# In[8]:


# avergae words per sentence 
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)

# Function to count the words
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

#Based on the dictionary of Loughran and McDonald (2016)
# Calculating positive score 
def positive_word_LM(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_LM:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_word_LM(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_LM:
            numNegWords -=1
    sumNeg = numNegWords 
    
    sumNeg = sumNeg * -1
    return sumNeg


# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / (positiveScore + negativeScore) 
    return pol_score

def ESG_percentage(text):
        numESGWords=0
        rawToken = tokenizer(text)
        for word in rawToken:
            if word in ESG_dict:
                numESGWords +=1
        
        sumESG = numESGWords 
        return sumESG

# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
     


# In[9]:


url = 'https://www.sec.gov/Archives/edgar/data/1318605/000156459017003118/tsla-10k_20161231.htm'


# ### Raw financial statement

# In[10]:


res = requests.get(url)
html = res.text
Tesla2017_Uncleaned = BeautifulSoup(html, 'html.parser')


# In[11]:


Tesla2017_Uncleaned


# #### Cleaning 1

# In[12]:


#Function to clean the dataset
def url_to_clean_text_round_1(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()
    text = soup.get_text()    
    
    
    # convert to lower case
    text = text.lower()
    text = re.sub(r'(\t|\v)', '', text)
    # remove \xa0 which is non-breaking space from ISO 8859-1
    text = re.sub(r'\xa0', ' ', text)
    # remove newline feeds (\n) following hyphens
    text = re.sub(r'(-+)\n{2,}', r'\1', text)
    # remove hyphens preceded and followed by a blank space
    text = re.sub(r'\s-\s', '', text)
    # replace 'and/or' with 'and or'
    text = re.sub(r'and/or', r'and or', text)
    # tow or more hypens, periods, or equal signs, possiblly followed by spaces are removed
    text = re.sub(r'[-|\.|=]{2,}\s*', r'', text)
    # all underscores are removed
    text = re.sub(r'_', '', text)
    # 3 or more spaces are replaced by a single space
    text = re.sub(r'\s{3,}', ' ', text)
    # three or more line feeds, possibly separated by spaces are replaced by two line feeds
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # remove hyphens before a line feed
    text = re.sub(r'-+\n', '\n', text)
    # replace hyphens preceding a capitalized letter with a space
    text = re.sub(r'-+([A-Z].*)', r' \1', text)
    # remove capitalized or all capitals for the months
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)', '', text)
    # remove years
    text = re.sub(r'2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019', '', text)
    # remove words million and company 
    text = re.sub(r'million|company', '', text)  
    # remove line feeds
    text = re.sub('\n', ' ', text)
    #replace single line feed \n with single space
    text = re.sub(r'\n', ' ', text)
    return text


# In[13]:


stage1 = url_to_clean_text_round_1(url)


# In[14]:


stage1


# #### Cleaning 2

# In[15]:


# Tokenizeing module and filtering tokens using stop words list, removing punctuations
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens))
    return filtered_words


# In[16]:


stage2 = tokenizer(stage1)


# In[17]:


stage2


# In[18]:


def textual_analysis(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    for table in soup.find_all('table'):
        table.decompose()
    text = soup.get_text()    
    
    
    # convert to lower case
    text = text.lower()
    text = re.sub(r'(\t|\v)', '', text)
    # remove \xa0 which is non-breaking space from ISO 8859-1
    text = re.sub(r'\xa0', ' ', text)
    # remove newline feeds (\n) following hyphens
    text = re.sub(r'(-+)\n{2,}', r'\1', text)
    # remove hyphens preceded and followed by a blank space
    text = re.sub(r'\s-\s', '', text)
    # replace 'and/or' with 'and or'
    text = re.sub(r'and/or', r'and or', text)
    # tow or more hypens, periods, or equal signs, possiblly followed by spaces are removed
    text = re.sub(r'[-|\.|=]{2,}\s*', r'', text)
    # all underscores are removed
    text = re.sub(r'_', '', text)
    # 3 or more spaces are replaced by a single space
    text = re.sub(r'\s{3,}', ' ', text)
    # three or more line feeds, possibly separated by spaces are replaced by two line feeds
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)
    # remove hyphens before a line feed
    text = re.sub(r'-+\n', '\n', text)
    # replace hyphens preceding a capitalized letter with a space
    text = re.sub(r'-+([A-Z].*)', r' \1', text)
    # remove capitalized or all capitals for the months
    text = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)', '', text)
    # remove years
    text = re.sub(r'2000|2001|2002|2003|2004|2005|2006|2007|2008|2009|2010|2011|2012|2013|2014|2015|2016|2017|2018|2019', '', text)
    # remove words million and company 
    text = re.sub(r'million|company', '', text)  
    # remove line feeds
    text = re.sub('\n', ' ', text)
    #replace single line feed \n with single space
    text = re.sub(r'\n', ' ', text)
 

    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
        

# count the words
    tokens = tokenizer(text)
    
# count the positive words  
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in pos_dict_LM:
            numPosWords  += 1
    
    sumPos = numPosWords
    
# Calculating Negative score
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in neg_dict_LM:
            numNegWords -=1
    sumNeg = numNegWords 
    
    sumNeg = sumNeg * -1  


        
    return round(average_sent_length), len(tokens),sumPos, sumNeg, url


# In[19]:


Textual_analysis_output = textual_analysis(url)


# In[20]:


Textual_analysis_output

