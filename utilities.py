# Import authentication stuffs
from auth import consumer_key, consumer_secret, access_token, access_token_secret

import tweepy
import tweepy
import pickle

import numpy as np

import datetime


def get_user_lat_long(api, geolocator, username):
    ''' Takes in a tweepy api, a geolocator object and a twitter username and
    returns the (latitude, longitude) tuple of the user's location if it
    exists. If not, (None, None) is returned '''

    longitude = None
    latitude = None
    try:
        user = api.get_user(username)
    except:
        return (latitude, longitude)

    if (user is not None):
        raw_location = user.location
        exit
        try:
            location = geolocator.geocode(raw_location)
        except:
            return (latitude, longitude)

        if (location is not None):
            longitude = location.longitude
            latitude = location.latitude

    return (latitude, longitude)


def extract_locations_from_file(api, geolocator, df, num_locs=None):
    """ 
    Extracts all the user's and their latitude longitude and saves it to a
    pickle file 
    """
    usernames = df['user']
    lat_longs = []
    count = 0

    for user in usernames:
        print(100. * count / len(usernames), "% complete")
        if num_locs != None and count == num_locs:
            break
        lat_long = get_user_lat_long(api, geolocator, user)
        lat_longs.append((user, lat_long))
        count += 1

    pickle.dump(lat_longs, open("lat_long_2017.p", "wb"))


def create_api():
    """
    Creates an api object that allows to scrape tweets from twitter
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api
