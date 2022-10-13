#import dependencies
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words


# force output to display the full description
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('filtered-data/eur-0-20.csv')
df = df[df['User Rating']>4]
df = df.reset_index()


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
stop_words = stopwords.words('english')
stop_words.extend(['wine', 'vin', 'like', 'bit', 'wife', 'nose', 'quite', 'note', 'red', 'nice', 'good', 'great', 'well', 'taste', 'really'])

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
        word) for word in split_text if not word in stop_words and len(word) > 2]
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
kmeans = KMeans(n_clusters=3)
# fit the data
kmeans.fit(vec_text)
# this loop transforms the numbers back into words
common_words = kmeans.cluster_centers_.argsort()[:, -1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# add the cluster label to the data frame
df['cluster'] = kmeans.labels_

#df.reset_index()

cdf = df.groupby('cluster')['User Rating'].mean()


print(cdf)

