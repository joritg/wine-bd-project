#import dependencies
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
import sqlite3
from sqlite3 import Error
from langdetect import detect


# force output to display the full description
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('filtered-data/eur-20-70.csv')


# add a column for the word count
df['word_count'] = df['Note'].apply(lambda x: len(str(x).split(" ")))
#print("Word Count Median: " + str(df['word_count'].median()))
# print(df['word_count'].describe())
x = df['word_count']
n_bins = 95
'''
plt.hist(x, bins=n_bins)
plt.xlabel('Number of Words in Review')
plt.ylabel('Frequency')
plt.show()
'''


# create a list of stop words
stop_words = set(stopwords.words("english"))



# show how many words are in the list of stop words
# print(len(stop_words))
# 179
# construct a new list to store the cleaned text
clean_desc = []
for w in range(len(df['Note'])):
    desc = str(df['Note'][w]).lower()

    # remove punctuation
    desc = re.sub('[^a-zA-Z]', ' ', desc)

    # remove tags
    desc = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", desc)

    # remove special characters and digits
    desc = re.sub("(\\d|\\W)+", " ", desc)

    split_text = desc.split()

    # Lemmatisation
    lem = WordNetLemmatizer()
    split_text = [lem.lemmatize(
        word) for word in split_text if not word == "wine" and not word == "nice" and not word == "good" and not word == "great" and not word == "well" 
        and not word == "taste" and not word == "really" and not word in stop_words and len(word) > 2]
    split_text = " ".join(split_text)
    clean_desc.append(split_text)


# add column to flag records with rating greater than 88
df['above_twenty'] = [1 if float(price) > 20.0 else 0 for price in df['price']]
df['above_avg'] = [1 if rating > 3.68 else 0 for rating in df['User Rating']]


# TF-IDF vectorizer
tfv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1))
# transform
vec_text = tfv.fit_transform(clean_desc)
# returns a list of words.
words = tfv.get_feature_names_out()

# setup kmeans clustering
kmeans = KMeans(n_clusters=15)
# fit the data
kmeans.fit(vec_text)
# this loop transforms the numbers back into words
common_words = kmeans.cluster_centers_.argsort()[:, -1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# add the cluster label to the data frame
df['cluster'] = kmeans.labels_
cdf = df.groupby('cluster')['price','User Rating'].mean()

print(cdf)


'''
clusters = df.groupby(['cluster', 'User Rating']).size()
fig, ax1 = plt.subplots(figsize=(26, 15))
sns.heatmap(clusters.unstack(level='price'), ax=ax1, cmap='Reds')
ax1.set_xlabel('price').set_size(18)
ax1.set_ylabel('cluster').set_size(18)
clusters = df.groupby(['cluster', 'above_twenty']).size()
fig2, ax2 = plt.subplots(figsize=(30, 15))
sns.heatmap(clusters.unstack(level='above_twenty'), ax=ax2, cmap="Reds")
ax2.set_xlabel('Above 20 euros').set_size(18)
ax2.set_ylabel('Cluster').set_size(18)
plt.show()
'''

'''
clusters = df.groupby(['cluster', 'User Rating']).size()
fig, ax1 = plt.subplots(figsize=(26, 15))
sns.heatmap(clusters.unstack(level='User Rating'), ax=ax1, cmap='Reds')
ax1.set_xlabel('User Rating').set_size(18)
ax1.set_ylabel('cluster').set_size(18)
clusters = df.groupby(['cluster', 'above_avg']).size()
fig2, ax2 = plt.subplots(figsize=(30, 15))
sns.heatmap(clusters.unstack(level='above_avg'), ax=ax2, cmap="Reds")
ax2.set_xlabel('Above Average Rating').set_size(18)
ax2.set_ylabel('Cluster').set_size(18)
plt.show()
'''

'''
#create dataframe of reviews not above average
not_above = df.loc[df['above_avg'] == 0]
not_above.describe()
#create data frame of reviews above average
above_avg = df.loc[df['above_avg'] == 1]
above_avg.describe()
#plot the counts
plt.figure(figsize=(14,4))
sns.countplot(x='cluster', data=not_above).set_title("Rating Counts")
plt.show()
plt.figure(figsize=(14,4))
sns.countplot(x='cluster', data=above_avg).set_title("Rating Counts")
plt.show()
'''
