#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Twitter text analysis

This script is meant for part 3 of our overall Twitter analysis.
It will be focusing on prepping and analyzing the dataset to answer the following questions:

1) What is the frequency of most occurring words?
2) What is the tone of general conservations?
3) What is the sentiment analysis?

# download wordnet
python -m nltk.downloader -d /Users/ktl014/miniconda3/envs/metoo_env/nltk_data wordnet

"""
from __future__ import print_function

import os
import random

# Third party imports
from multiprocessing import Pool
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project level imports

# Module level constants
CORPUS_FNAME = 'datasets/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

# ============== BEGIN: Cleaning Tweet ============== #
def clean_tweet(tweet):
    '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
    '''
    import re
    import string
    # Initialize stop words
    stop_words = set(stopwords.words("english"))

    # Remove symbols and speical characters
    tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w+:\ / \ / \S+)", " ", tweet)

    # Remove links
    tweet = ' '.join(word for word in tweet.split(' ') if not (
                re.match(r"http", word) or re.match(r".*\..*/", word)
                or re.match(r"#", word) or re.match(r"\d+", word) or (len(word) == 1)))
    # Strip punctuation
    tweet = tweet.translate(string.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    #Remove Emojis
    tweet = remove_emoji(tweet)
    # Take out non-stopwords
    tweet = [w.lower() for w in tweet.split(' ') if not w in stop_words]
    # Join as string
    tweet = ' '.join(w for w in tweet)
    return tweet

def remove_emoji(text):
    """Remove Emojis"""
    import emoji
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

# ============== END: Cleaning Tweet ============== #
#
#
#
#
# ============== BEGIN: Coroutines ============== #
def percentage_coroutine(to_process, print_on_percent = 0.05):
    """Percentage monitor coroutine"""
    print ("Starting progress percentage monitor")

    processed = 0
    count = 0
    print_count = to_process*print_on_percent
    while True:
        yield
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed)/float(to_process))*100

            print("{:.0f}% finished".format(pct))

def trace_progress(func, progress = None):
    def callf(*args, **kwargs):
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf

def my_func(i):
    return i ** 2

# ============== END: Coroutines ============== #
#
#
#
#
# ============== BEGIN: Plotting ============== #
def plot_sentiment(data, year=17):
    """Plot histogram of sentiment affects"""
    df = data.copy()
    nrc = NRCLexicon()
    temp = df[nrc.sentiments].sum().sort_values()
    frequency = temp.values
    labels = temp.index.to_list()
    N = len(frequency)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6))
    rect = ax.bar(ind, frequency, width=width, align='center')
    ax.set_xticks(ind)
    ax.set_xticklabels(tuple(labels))
    ax.tick_params(axis='x', rotation='auto')
    ax.set_xlabel('Sentiment')
    plt.ylabel('Frequency')

    for bar in rect:
        yval = bar.get_height()
        ax.text(bar.get_x()+0.05, yval + .035, yval, fontsize=12)

    plt.tight_layout()
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/sentiment_{}.png'.format(year))
    plt.show()

# ============== END: Plotting ============== #
#
#
#
#
# ============== BEGIN: NRC Mapping ============== #
def lemmed(text, cores=6): # tweak cores as needed
    with Pool(processes=cores) as pool:
        wnl = WordNetLemmatizer()
        result = pool.map(wnl.lemmatize, text)
    return result

class NRCLexicon(object):
    """ NRC Lexicon Parsing, Mapping and Scoring

    Handy instance to help with conducting sentiment analysis.

    """
    TG = 'target_word'
    AFF = 'affect'
    AFF_CAT = 'affect_cat'
    AF = 'association_flag'

    def __init__(self, fname=CORPUS_FNAME):
        """Initializes NRCLexicon"""
        self.nrc = pd.read_csv(fname, sep='\t',
                          names=[self.TG, self.AFF, self.AF])
        self.nrc[self.AFF_CAT] = self.nrc[self.AFF].astype('category').cat.codes
        self.sentiments = sorted(set(self.nrc[self.AFF]))

    def _is_word_available(self, word):
        """Checks if word is available within corpus"""
        return self.nrc[self.TG].str.contains(word).any()


    def sentiment_score(self, query):
        """Returns sentiment score given a list of words"""
        query = nltk.pos_tag(query.split())
        nouns_only = []
        for i,(word, pos) in enumerate(query):
            try:
                word = WordNetLemmatizer().lemmatize(word, 'n')
                nouns_only.append(word)
            except:
                pass

        df = self.nrc[self.nrc[self.TG].isin(nouns_only) & self.nrc[self.AF] == 1].reset_index(
            drop=True)
        counts = df['affect_cat'].value_counts().sort_index()
        counts = counts.reindex(np.arange(0, 10)).fillna(0)
        counts = np.array(counts.to_list())
        return counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6], \
               counts[7], counts[8], counts[9]
# ============== END: NRC Mapping ============== #
#
#
#
#
if __name__ == '__main__':
    dataset_dir = 'datasets'
    df = pd.DataFrame()
    for data in os.listdir(dataset_dir):
        if data.endswith('.csv'):
            # Read in data
            df = pd.read_csv(os.path.join(dataset_dir, data))
            # smpl_size = 1
            # df = df.sample(smpl_size).reset_index(drop=True) # UNCOMMENT FOR QUICK DEV

            # Format into sentiment dataframe
            nrc = NRCLexicon()
            df = pd.concat([df, pd.DataFrame(columns=nrc.sentiments).fillna(0)], sort=False)

            # Clean tweet
            co = percentage_coroutine(len(df))
            print('Cleaning tweets')
            df['text'] = df['text'].apply(trace_progress(clean_tweet, progress=co))

            # Take sentiment score
            print('Taking sentiment score')
            co = percentage_coroutine(len(df))
            for ii, row in df.iterrows():
                if ii % 5000 == 0:
                    print('{}/{}'.format(ii, len(df)))
                row = row.copy()
                df.loc[ii, nrc.sentiments] = nrc.sentiment_score(row['text'])

            #TODO functions to debug below
            # df[nrc.sentiments] = df['text'].apply(nrc.sentiment_score, axis=1)
            # df[nrc.sentiments] = df['text'].apply(trace_progress(nrc.sentiment_score, progress=co))

            # Plot sentiment
            plot_sentiment(data=df)

    """Single usage"""
    # nrc = NRCLexicon()
    # df = df.sample(1).reset_index(drop=True)
    # df = pd.concat([df,pd.DataFrame(columns=nrc.sentiments).fillna(0)], sort=False)
    # query = clean_tweet(df['text'][0])
    #
    # stopwords_set = set(stopwords.words("english"))
    #
    # words_filtered = [e.lower() for e in query.split() if len(e) >= 3]
    # words_cleaned = [word for word in words_filtered
    #                  if 'http' not in word
    #                  and not word.startswith('@')
    #                  and not word.startswith('#')
    #                  and word != 'RT']
    # words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    # df[nrc.sentiments] = nrc.sentiment_score(words_without_stopwords)
    #
    # plot_sentiment(data=df)
