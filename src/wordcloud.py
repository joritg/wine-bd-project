# Importing modules
import pandas as pd
# Load the regular expression library
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint


# Import the wordcloud library
from wordcloud import WordCloud

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import words

# Read data into papers
df = pd.read_json('./filtered-data/eur-50+.csv')

# Remove the columns
df = df.drop(columns=['Year','Wine ID','CreatedAt','Winery','Wine','Rating','num_review','language'], axis=1)


# create a list of stop words
stop_words = stopwords.words('english')
stop_words.extend(['wine', 'vin', 'palate', 'red', 'nice', 'good', 'great', 'well', 'taste', 'really','nose','note'])

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
    df['Note'][w] = split_text


# Join the different processed titles together.
long_string = ','.join(list(df['Note'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()
wordcloud.to_file('cloud-50+.png')

