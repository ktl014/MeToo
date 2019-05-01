import tweepy

# Import authentication stuffs
from auth import consumer_key, consumer_secret, access_token, access_token_secret

import pickle


def create_api():
    """
    Creates an api object that allows to scrape tweets from twitter
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def get_metoo_tweets(api, num_tweets=50000):
    """
    Extracts num_tweets tweets with the #metoo hashtag from twitter using
    tweepy's api and returns a list of tweet objects
    """
    i = 1
    tweets = []
    for tweet in tweepy.Cursor(api.search, q='#metoo', lang="en").items(num_tweets):
        # print(i)
        # print(tweet.user.screen_name + " tweeted: ")
        # print(tweet.text)
        tweets.append(tweet)
        i += 1

    return tweets


def unpickle_metoo_tweets(tweet_file):
    """
    Reads in the pickled tweets found in tweet_file and returns a list of tweet
    objects so we don't have to rescrape them
    """
    f = open(tweet_file, "rb")
    metoo_tweets = pickle.load(f)

    return metoo_tweets


def main():
    # First create the api to begin scraping
    api = create_api()

    # File name for storing and loading tweets
    tweet_file = "pickled_tweets.p"

    # Scrape 50,000 #metoo tweets
    # tweets = get_metoo_tweets(api, num_tweets=50000)

    # Save tweets to a pickle file for later reading
    # f = open(tweet_file, "wb")
    # pickle.dump(tweets, f)

    # Read back tweets from pickle file
    metoo_tweets = unpickle_metoo_tweets(tweet_file)


if __name__ == '__main__':
    main()
