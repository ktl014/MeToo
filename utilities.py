# Import authentication stuffs
from auth import consumer_key, consumer_secret, access_token, access_token_secret
import tweepy
import tweepy
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import math
import os

import requests

from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure


import imageio

# From https://stackoverflow.com/questions/20169467/how-to-convert-from-longitude-and-latitude-to-country-or-city
data = requests.get(
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

countries = {}
for feature in data["features"]:
    geom = feature["geometry"]
    country = feature["properties"]["ADMIN"]
    countries[country] = prep(shape(geom))


def get_country(lon, lat):
    point = Point(lon, lat)
    for country, geom in countries.items():
        if geom.contains(point):
            return country

    return "unknown"


def get_retweet_by_user_date(date_retweets_dict, user, date):
    retweet_cnt_user = date_retweets_dict[date]

    for retweet_user in retweet_cnt_user:
        if(retweet_user[0] == user):
            return retweet_user[1]


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
    dates = df['date']

    if(num_locs == None):
        num_locs = len(usernames)

    lat_longs = []
    count = 0

    for i in range(len(usernames)):
        user = usernames[i]
        date = dates[i]

        print(100. * count / num_locs, "% complete")
        if count == num_locs:
            break

        lat_long = get_user_lat_long(api, geolocator, user)
        lat_longs.append((user, lat_long, date))
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


def plot_locations(lat_longs, plot_us=True, date_retweets_dict=None, over_time=True, save_every=None, batch_size=100, color_gradient=False):
    if(plot_us):
        filename = "images/us/"
        scalar = .5
        fig = plt.figure(figsize=(40, 30))
    else:
        filename = "images/global/"
        scalar = 0.5
        fig = plt.figure(figsize=(40, 20))

    if(plot_us):
        # Create map of US
        m = Basemap(projection="lcc", lat_1=33, lon_0=-95,
                    llcrnrlat=22, urcrnrlat=49,
                    llcrnrlon=-119, urcrnrlon=-64, resolution='i', area_thresh=10000)

        m.drawstates(linewidth=0.5)
        m.drawcoastlines(linewidth=0.5)
        m.drawcountries(linewidth=0.5)
    else:
        # Create map of World

        m = Basemap(projection="mill",  # lat_1=33, lon_0=-95,
                    llcrnrlat=-50, urcrnrlat=70,
                    llcrnrlon=-180, urcrnrlon=180, resolution='i')

        m.drawcoastlines(linewidth=0.2)
        m.drawcountries(linewidth=0.5)

    count = 1
    while(count + batch_size <= len(lat_longs)):
        current_batch = lat_longs[count:count+batch_size]

        lats = []
        lons = []
        dates = []
        retweet_cnts = []

        for datapoint in current_batch:
            user = datapoint[0]
            lon = datapoint[1][0]
            lat = datapoint[1][1]
            date = datapoint[2]

            lats.append(lat)
            lons.append(lon)

            dates.append(date)
            retweet_cnts.append(get_retweet_by_user_date(
                date_retweets_dict, user, date) * scalar)

        if(color_gradient):
            # Color and size change
            m.scatter(lats, lons, latlon=True, marker='o',
                      c=retweet_cnts, s=retweet_cnts, cmap=plt.get_cmap('seismic'), alpha=.1)
        else:
            # Only size changes
            m.scatter(lats, lons, latlon=True, marker='o',
                      c='r', s=retweet_cnts, cmap=plt.get_cmap('seismic'), alpha=.1)
        plt.pause(0.0001)

        if(save_every is not None):
            if(count % save_every == 0):
                plt.savefig(filename + str(count).zfill(5) + ".png")
        print(date)
        count += batch_size
    # plt.show()


def filter_lat_longs(lat_longs, us_only=False):
    filtered_lat_longs = []

    for lat_long in lat_longs:
        if lat_long[1][0] is not None:
            # If we only want US lat, longs, then we need to check which country they are in
            if(us_only and ("America" in get_country(lat_long[1][1], lat_long[1][0]))):
                filtered_lat_longs.append(lat_long)
            elif (not us_only):
                filtered_lat_longs.append(lat_long)

    return filtered_lat_longs


def generate_gif(directory):
    images = []
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    print(filenames)
    for filename in filenames:
        if(".png" in filename):
            images.append(imageio.imread(directory + filename))

    imageio.mimsave(directory + "/video.gif", images,
                    format='GIF', duration=.1)


def display_gif_bokeh():
    pass

def freq_dict(file_list,noun_type='all',most_common_no=100):
    '''Returns list of tuples of string to number, i.e., noun to number of occurences, for tweets stored in specified file.
    Can show proper or common nouns selectively.
    Default number of most frequent names/nouns = 100'''
    assert (noun_type=='all' or noun_type=='proper'), 'Only \'proper\' and \'all\' options allowed'
    #import stuff
    import string
    import re
    import pandas as pd
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    #if only one file specified:
    if isinstance(file_list,str):
        file_list = list(file_list)
    ret_list = []
    for file_name in file_list:
        df = pd.read_csv(file_name)
        tweet_texts = df['text']
        for tweet in tweet_texts:
            orig_tweet = tweet
            #clean tweet text
            tweet = re.sub(r"â€|Ã©|™|¦|œ|", "", tweet)
            tweet = ' '.join(word for word in tweet.split(' ') if not (re.match(r"http",word) or re.match(r".*\..*/",word) or re.match(r"#",word) or re.match(r"\d+",word) or (len(word)==1)))
            tweet = tweet.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            tweet = [w for w in tweet.split(' ') if not w in stop_words]
            tweet = ' '.join(w for w in tweet)
            if noun_type=='proper':
                prop_noun_string = pos_tag(tweet.split())
                proper_nouns = [word for word,pos in prop_noun_string if pos == 'NNP']
                fake_proper_nouns = ['Women','Im','Thank','Please','Sexual','New','Movement','My','How','No','NOT','Men','Me','RT','THE','Hey','Year','Dont','TIME','â\x81','YOU','Are','Do','Join','Harassment','U','Her','News','Judge','Too','Rape','Times','State','ALL','Remember','March','Ive','Just','Assault','Time','IT','Let','Read','TO','Has','Up','Be','So','Person','THIS','Did','Twitter','NO','Media','R','Stop','Cold','Where','Watch','Dear','Baby','Street','Police','Today','South','Against','Outside','Heard','Justice','Check','Great','Or','All','Fake','Ill','Ad','Listen','Green','IN','Act','Christmas','Singer','Well','Who','ON','Day','NOW','Wow','Get','University','Man','TheRestlessQuil','NBC']
                ret_list.extend([word for word in proper_nouns if not word in fake_proper_nouns])
            else:
                tokens = word_tokenize(tweet.lower())
                is_noun = lambda pos: pos[:2] == 'NN'
                nouns = [word for (word, pos) in nltk.pos_tag(tokens) if is_noun(pos)]
                #ret_list.extend([word for word in nouns]) #FOR ACTUAL NOUNS
                #FOR ROOT WORD OF NOUN ONLY
                stemmed = [porter.stem(word) for word in nouns]
                #extend list of encountered words
                ret_list.extend([word for word in stemmed])
    c = Counter(ret_list)
    #return counter object of most common words
    return (c.most_common(most_common_no))
