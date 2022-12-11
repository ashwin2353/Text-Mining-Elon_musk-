# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:06:56 2022

@author: ashwi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
# pip install -U textblob
from textblob import TextBlob
from wordcloud import WordCloud
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
#================================================================================
df = pd.read_csv("Elon_musk.csv",encoding="latin-1")
df
df["Text"]
stp_words = stopwords.words("english")
stp_words 

##########################   sentimental analysis #################################
# cleaning the tweets

one_tweet = df.iloc[5]['Text']

def TweetCleaning(tweets):
    Cleantweet=re.sub(r"@[a-zA-Z0-9]+"," ",tweets)
    Cleantweet=re.sub(r"#[a-zA-Z0-9]+"," ",Cleantweet)
    Cleantweet=''.join(word for word in Cleantweet.split() if word not in stp_words)
    return Cleantweet

def calPolarity(tweets):
    return TextBlob(tweets).sentiment.polarity

def calSubjectivity(tweets):
    return TextBlob(tweets).sentiment.subjectivity

def segmentation(tweets):
    if tweets > 0:
        return "positive"
    elif tweets == 0:
        return "neutral"
    else:
        return "negative"

df["Cleanedtweets"]=df["Text"].apply(TweetCleaning)
df["polarity"]=df["Cleanedtweets"].apply(calPolarity)
df["subjectivity"]=df["Cleanedtweets"].apply(calSubjectivity)
df["segmentation"]=df["polarity"].apply(segmentation)

pd.set_option("display.max_columns",6)
df.head()

##########################   Analysis and visualization ################################

df.pivot_table(index=['segmentation'],aggfunc={"segmentation":'count'})

# The positive tweets are 82

# The negative tweets are 3

# The neutral tweets are 1914

# Top three positive tweets
df.sort_values(by=['polarity'],ascending=False).head(3)
 
# Top three nagative tweets
df.sort_values(by=['polarity'],ascending=True).head(3)

# Top three neutral tweets

df['polarity']==0
df[df['polarity']==0].head(3)

#======================================================================================
#################################  Text Preprocessing  ###############################
df["Cleanedtweets"]

# Joining the list into one string/text

text = ' '.join(df["Cleanedtweets"])
text

# punctuation
no_punc_text = text.translate(str.maketrans('','',string.punctuation))
no_punc_text

# Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])

# Removeing stopwords
my_stop_words = stopwords.words("english")

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
no_stop_tokens

# Normaliza the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:40])

# Stemming the data
from nltk.stem import PorterStemmer
ps=PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:10])

#===========================================================================
import spacy

nlp = spacy.load("en_core_web_sm")
# lemmas being one of them, but mostly POS, which whill will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])

lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])

############################# Feature Extraction  ###############################

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_tokens)

pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)

print(vectorizer.vocabulary_)

print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])

print(X.toarray().shape)

# bigram and trigram

vectorizer_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=(100))
bow_matrix_ngram = vectorizer_ngram_range.fit_transform(df["Cleanedtweets"])

bow_matrix_ngram 

print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# TFidf vectorizer 

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_range_max_features = TfidfVectorizer(analyzer="word",ngram_range=(1,3),max_features=10)
tf_idf_matrix_n_gram_max_features = vectorizer_n_gram_range_max_features.fit_transform(df["Cleanedtweets"])

print(vectorizer_n_gram_range_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())

# wordcloud
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud,STOPWORDS
# Define a function to plot word cloud

def plot_cloud(wordcloud):
    plt.figure(figsize=(15,30))
    plt.imshow(wordcloud)
    plt.axis("off");

wordcloud = WordCloud(width= 3000, height=2000, background_color="black", max_words=100, colormap="Set2").generate(text)
# plot
plot_cloud(wordcloud)












