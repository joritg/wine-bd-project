# Importing modules
import pandas as pd
import json
# Load the regular expression library
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


df = pd.read_csv('./filtered-data/eur-50+.csv')


stop_words = stopwords.words('english')
stop_words.extend(['wine', 'vin', 'palate', 'red', 'nice', 'good', 'great', 'well', 'taste', 'really','nose','note','one'])
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
    tokens = word_tokenize(split_text)
    df['Note'][w] = tokens



#create dictionary
text_dict = Dictionary(df['Note'])

#view integer mappings

tweets_bow = [text_dict.doc2bow(tweet) for tweet in df['Note']]

k = 5
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)

print(tweets_lda.show_topics())


vis = pyLDAvis.gensim_models.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word)
pyLDAvis.save_html(vis, 'vivino_vis-50+.html')
