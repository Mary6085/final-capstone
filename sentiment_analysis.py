# Capstone project.
import numpy as np
import pandas as pd
import spacy

amazon = pd.read_csv('datasets/Datafiniti_Amazon_consumer_Reviews_of_Amazon_Products.csv')
amazon.head()

# Cleaning of the dataset.
cleaned = amazon[['reviews.text', 'reviews.text', 'reviews.username']]
cleaned

# Removing the Nan/Null values
cleaned.isnull().sum()

cleaned.dropna(inplace=True, axis=0)
cleaned.isnull().sum()

# Creating a function to preprocess the data and a function for sentiment analysis
text = cleaned['reviews.text']
text

nlp = spacy.load('en_core_web_sm')

def preprocess(text):

    doc = nlp(text.lower().strip())
    processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]  

    return ''.join(processed)

cleaned['processed.text'] = cleaned['reviews.text'].apply(preprocess)
cleaned.head()   # Preprocessed text/data

# Polarity test
import textblob
import wordcloud
import matplotlib.pyplot as plt

from collections import defaultdict

nlp = spacy.load('en_core_web_sm')

positive_words = defaultdict(int)
negative_words = defaultdict(int)

# working on the sentiment analysis
for sentence in cleaned['processed.text']:
    doc = nlp(sentence)
    tokens = [token.lemma_.lower().strip() for token in doc if not token.is_stop and token.is_alpha]
for token in tokens:
    blob = TextBlob(str(token))

    polarity = blob.sentiment.polarity

    if polarity > 0:
        positive_words[token.lower()] += 1
    elif polarity < 0:
        positive_words[token.lower()] += 1

pos_wordcloud = wordcloud(width=600,height=400,background_color = 'white').generate_from_frequencies(positive_words)
neg_wordcloud = wordcloud(width=600,height=400,background_color = 'white').generate_from_frequencies(negative_words)

fig, ax = plt.subplot(1, 2, figsize=(10,5))

ax[0].imshow(pos_wordcloud, interpolation='bilinear')
ax[0].set_title('positive words')
ax[0].axis('off')

ax[0].imshow(neg_wordcloud, interpolation='bilinear')
ax[0].set_title('negative words')
ax[0].axis('off')

plt.show()