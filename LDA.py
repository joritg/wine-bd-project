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



with open('stream.json') as json_file:
        tweet_dict = json.load(json_file)

data = []
for v in tweet_dict.values():
    data.append(v)

data = list(set(data))

df = pd.DataFrame(data)
df.rename( columns={0 :'text'}, inplace=True )

df['text'] = df['text'].str.replace(r"http\S+", "")
df['text'] = df['text'].str.replace(r"@\S+", "")


stop_words = stopwords.words('english')
stop_words.extend(['like', 'bobi', 'amp', 'know', 'get', 'let', 'would', 'make', 'say', 'mean', 'mukulu', 'uganda', 'ugandaisbleeding', 'mayhemtilthemorning', 'ugandan'])
clean_desc = []
for w in range(len(df['text'])):
    desc = str(df['text'][w]).lower()

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
    df['text'][w] = tokens



#create dictionary
text_dict = Dictionary(df.text)

#view integer mappings

tweets_bow = [text_dict.doc2bow(tweet) for tweet in df['text']]

k = 2
tweets_lda = LdaModel(tweets_bow,
                      num_topics = k,
                      id2word = text_dict,
                      random_state = 1,
                      passes=10)

print(tweets_lda.show_topics())


vis = pyLDAvis.gensim_models.prepare(tweets_lda, tweets_bow, dictionary=tweets_lda.id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
