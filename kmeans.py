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
import sqlite3
from sqlite3 import Error
from langdetect import detect


# force output to display the full description
pd.set_option('display.max_colwidth', None)

'''
#connect to database file
conn = sqlite3.connect('db\wine_data.sqlite')
c = conn.cursor()
#create dataframe from sql query
df = pd.read_sql("Select country, description, rating, price, title, variety from wine_data where variety = 'Chardonnay'", conn)
#display the top 3 rows
df.head(3)
'''

'''
def get_wine_data(wine_id, year, page):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    }

    api_url = "https://www.vivino.com/api/wines/{id}/reviews?per_page=50&year={year}&page={page}"  # <-- increased the number of reviews to 9999

    data = requests.get(
        api_url.format(id=wine_id, year=year, page=page), headers=headers
    ).json()

    return data


r = requests.get(
    "https://www.vivino.com/api/users/",
    params={
        "country_codes[]": ["pt","es","it","fr"],
        "language": "english",
        "currency_code": "EUR",
        "grape_filter": "varietal",
        "min_rating": "1",
        "order_by": "price",
        "order": "asc",
        "page": 1,
        "price_range_max": "30",
        "price_range_min": "20",
        "wine_type_ids[]": "1",
    },
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0"
    },
)

results = [
    (
        t["vintage"]["wine"]["winery"]["name"],
        t["vintage"]["year"],
        t["vintage"]["wine"]["id"],
        f'{t["vintage"]["wine"]["name"]} {t["vintage"]["year"]}',
        t["vintage"]["statistics"]["ratings_average"],
        t["vintage"]["statistics"]["ratings_count"],
    )
    for t in r.json()["explore_vintage"]["matches"]
]
dataframe = pd.DataFrame(
    results,
    columns=["Winery", "Year", "Wine ID", "Wine", "Rating", "num_review"],
)

ratings = []
for _, row in dataframe.iterrows():
    page = 1
    while True:
        print(
            f'Getting info about wine {row["Wine ID"]}-{row["Year"]} Page {page}'
        )

        d = get_wine_data(row["Wine ID"], row["Year"], page)

        if not d["reviews"]:
            break

        for r in d["reviews"]:
            ratings.append(
                [
                    row["Year"],
                    row["Wine ID"],
                    r["rating"],
                    r["note"],
                    r["created_at"],
                ]
            )

        page += 1

ratings = pd.DataFrame(
    ratings, columns=["Year", "Wine ID", "User Rating", "Review", "CreatedAt"]
)


df = ratings.merge(dataframe)
'''



df = pd.read_csv('sek-400kr FILTERED.csv')




# add a column for the word count
df['word_count'] = df['Note'].apply(lambda x: len(str(x).split(" ")))
#print("Word Count Median: " + str(df['word_count'].median()))
#print(df['word_count'].describe())
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
#print(len(stop_words))
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




#add column to flag records with rating greater than 88
df['above_avg'] = [1 if rating > 4 else 0 for rating in df['User Rating']]

# TF-IDF vectorizer
tfv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 1))
# transform
vec_text = tfv.fit_transform(clean_desc)
# returns a list of words.
words = tfv.get_feature_names_out()

#setup kmeans clustering
kmeans = KMeans(n_clusters = 10, n_init = 17, tol = 0.01, max_iter = 200)
#fit the data 
kmeans.fit(vec_text)
#this loop transforms the numbers back into words
common_words = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

#add the cluster label to the data frame
df['cluster'] = kmeans.labels_
clusters = df.groupby(['cluster', 'User Rating']).size()
fig, ax1 = plt.subplots(figsize = (26, 15))
sns.heatmap(clusters.unstack(level = 'User Rating'), ax = ax1, cmap = 'Reds')
ax1.set_xlabel('User Rating').set_size(18)
ax1.set_ylabel('cluster').set_size(18)
clusters = df.groupby(['cluster', 'above_avg']).size()
fig2, ax2 = plt.subplots(figsize = (30, 15))
sns.heatmap(clusters.unstack(level = 'above_avg'), ax = ax2, cmap="Reds")
ax2.set_xlabel('Above Average Rating').set_size(18)
ax2.set_ylabel('Cluster').set_size(18)
plt.show()

