"""Twitter text analysis

This script is meant for part 3 of our overall Twitter analysis.
It will be focusing on prepping and analyzing the dataset to answer the following questions:

1) What is the frequency of most occurring words?
2) What is the tone of general conservations?
3) What is the sentiment analysis?

# download wordnet
python -m nltk.downloader -d /Users/ktl014/miniconda3/envs/metoo_env/nltk_data wordnet

"""
# Standard dist imports
import os
import random

# Third party imports
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd

# Project level imports

# Module level constants
fname = 'datasets/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

def clean_tweet(tweet):
    '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
    '''
    import re
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\ / \ / \S+)", " ",
                           tweet).split()).lower()

class NRCLexicon(object):
    TG = 'target_word'
    AFF = 'affect'
    AFF_CAT = 'affect_cat'
    AF = 'association_flag'

    def __init__(self, fname=fname):
        self.nrc = pd.read_csv(fname, sep='\t',
                          names=[self.TG, self.AFF, self.AF])
        self.nrc[self.AFF_CAT] = self.nrc[self.AFF].astype('category').cat.codes
        self.sentiments = sorted(set(self.nrc[self.AFF]))

    def _is_word_available(self, word):
        return self.nrc[self.TG].str.contains(word).any()


    def sentiment_score(self, query):
        query = nltk.pos_tag(query)
        nouns_only = []
        for i,(word, pos) in enumerate(query):
            try:
                word = WordNetLemmatizer().lemmatize(word, 'n')
                nouns_only.append(word)
            except:
                print(word)
        df = self.nrc[self.nrc[self.TG].isin(nouns_only) & self.nrc[self.AF] == 1].reset_index(
            drop=True)
        counts = df['affect_cat'].value_counts().sort_index()
        counts = counts.reindex(np.arange(0, 10)).fillna(0)
        counts = np.array(counts.to_list())
        return counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], \
               counts[7], counts[8], counts[9]

if __name__ == '__main__':
    dataset_dir = 'datasets'
    df = pd.DataFrame()
    for data in os.listdir(dataset_dir):
        if data.endswith('.csv'):
            temp = pd.read_csv(os.path.join(dataset_dir, data))
            df = df.append(temp).reset_index(drop=True)

    nrc = NRCLexicon()
    df = df.sample(1).reset_index(drop=True)
    df = pd.concat([df,pd.DataFrame(columns=nrc.sentiments).fillna(0)], sort=False)
    query = clean_tweet(df['text'][0])

    stopwords_set = set(stopwords.words("english"))

    words_filtered = [e.lower() for e in query.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
                     if 'http' not in word
                     and not word.startswith('@')
                     and not word.startswith('#')
                     and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    df[nrc.sentiments] = nrc.sentiment_score(words_without_stopwords)
