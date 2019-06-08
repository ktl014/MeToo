# Import authentication stuffs
from auth import consumer_key, consumer_secret, access_token, access_token_secret
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


import imageio

from geopy.geocoders import Nominatim
import pandas as pd


def create_country_dict():
    ''' Generates a dictionary that maps countries to polygons '''
    data = requests.get(
        "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

    countries = {}
    for feature in data["features"]:
        geom = feature["geometry"]
        country = feature["properties"]["ADMIN"]
        countries[country] = prep(shape(geom))
    return countries


def get_country(lon, lat, countries):
    ''' Given a longitude latitude of a tweet, and the country dictionary,
    extracts the country name '''
    point = Point(lon, lat)
    for country, geom in countries.items():
        if geom.contains(point):
            return country

    return "unknown"


def get_retweet_by_user_date(date_retweets_dict, user, date):
    ''' Extracts a retweet given the user, date and a dictionary containing
    dates to retweets '''
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


def plot_locations(lat_longs, plot_us=True, date_retweets_dict=None, save_every=None, batch_size=100, color_gradient=False):
    ''' Plots a map of either US or International to visualize #metoo tweets. Scales each tweet by its retweet count'''
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
    # Go through each batch and plot
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
        # Uncomment for live viewing
        # plt.pause(0.0001)

        if(save_every is not None):
            if(count % save_every == 0):
                plt.savefig(filename + str(count).zfill(5) + ".png")

        # print(date)
        count += batch_size
    plt.show()


def filter_lat_longs(lat_longs, countries, us_only):
    ''' Filters latitude and longitudes to remove Nones and if only looking for US,
    only reture lat longs that are within US '''
    filtered_lat_longs = []

    for lat_long in lat_longs:
        if lat_long[1][0] is not None:
            # If we only want US lat, longs, then we need to check which country they are in
            if(us_only and ("America" in get_country(lat_long[1][1], lat_long[1][0], countries))):
                filtered_lat_longs.append(lat_long)
            elif (not us_only):
                filtered_lat_longs.append(lat_long)

    return filtered_lat_longs


def generate_gif(directory):
    ''' Generates a gif from the files in directory '''
    images = []
    filenames = os.listdir(directory)
    filenames = sorted(filenames)
    for filename in filenames:
        if(".png" in filename):
            images.append(imageio.imread(directory + filename))

    imageio.mimsave(directory + "/video.gif", images,
                    format='GIF', duration=.1)


<<<<<<< HEAD
def extract_retweet_numbers(df):
    ''' Extracts retweet numbers from dataframe '''
    retweet_cnts = df['retweets']
    dates = df['date']
    users = df['user']

    date_to_retweets_dict = {}

    for i in range(len(retweet_cnts)):
        if(dates[i]) not in date_to_retweets_dict:
            date_to_retweets_dict[dates[i]] = [(users[i], retweet_cnts[i])]
        else:
            date_to_retweets_dict[dates[i]].append((users[i], retweet_cnts[i]))

    return date_to_retweets_dict


def extract_locations():
    ''' Extracts user names from tweet csv file and then uses usernames
    to get their locations via tweepy'''
    # Create the tweepy api to extract username info (location)
    api = create_api()

    # Create the geolocator to extract latitude and longitude from user location
    geolocator = Nominatim(user_agent="metwoo_data_analysis")

    # Read in dataframe
    df = pd.read_csv('datasets/tweet_2017_to_2018_with_hashtags.csv')

    # Extract locations from each tweet
    extract_locations_from_file(api, geolocator, df, num_locs=100)


def get_dates_and_retweets():
    ''' Gets dates and their associated retweets from csv files to combine them into
    a single dictionary '''
    # Uncomment to extract locations from file (Otherwise, they will are in the repo)
    # extract_locations()
    df = pd.read_csv('datasets/tweet_2017_to_2018_with_hashtags.csv')
    date_retweets_2017 = extract_retweet_numbers(df)

    df = pd.read_csv('datasets/tweet_2018_to_2019_with_hashtags.csv')
    date_retweets_2018 = extract_retweet_numbers(df)

    date_retweets_dict = {**date_retweets_2017, **date_retweets_2018}

    return date_retweets_dict


def format_lat_longs(countries):
    ''' Reverses and combines lat longs, and extracts us only lat_longs
    to return '''
    lat_longs_2017 = pickle.load(open("lat_long_2017.p", "rb"))
    lat_longs_2018 = pickle.load(open("lat_long_2018_2019.p", "rb"))

    lat_longs_2017.reverse()
    lat_longs_2018.reverse()

    lat_longs_orig = lat_longs_2017 + lat_longs_2018

    lat_longs = filter_lat_longs(lat_longs_orig, countries, us_only=False)

    us_lat_longs = filter_lat_longs(lat_longs_orig, countries, us_only=True)

    return lat_longs, us_lat_longs


def get_retweets_by_country(lat_longs, date_retweets_dict, countries):
    ''' Extracts the number of retweets associated with each country '''
    tweet_by_country_dict = {}
    retweets_by_country_dict = {}

    # Detetmine the country of each lat long
    for lat_long in lat_longs:
        country = get_country(
            lon=lat_long[1][1], lat=lat_long[1][0], countries=countries)
        if country in tweet_by_country_dict:
            tweet_by_country_dict[country].append(lat_long)
        else:
            tweet_by_country_dict[country] = [lat_long]

    # For each country, determine the sum of retweets
    for country in tweet_by_country_dict:
        for tweet in tweet_by_country_dict[country]:
            retweets = get_retweet_by_user_date(
                date_retweets_dict, tweet[0], tweet[2])
            if(country in retweets_by_country_dict):
                retweets_by_country_dict[country] += retweets
            else:
                retweets_by_country_dict[country] = retweets

    return retweets_by_country_dict


def get_minimal_retweets_by_country(retweets_by_country_dict):
    ''' Gets a subset of retweets by country based on the retweet_cnt '''
    minimal_retweets_by_country_dict = {}

    minimal_retweets_by_country_dict["Other"] = retweets_by_country_dict.pop(
        "unknown")

    minimal_retweets_by_country_dict["United States"] = retweets_by_country_dict.pop(
        "United States of America")

    for country in retweets_by_country_dict:
        retweet_cnt = retweets_by_country_dict[country]
        if(retweet_cnt < 5000):
            minimal_retweets_by_country_dict["Other"] += retweet_cnt
        else:
            minimal_retweets_by_country_dict[country] = retweet_cnt

    sorted_minimal_retweets_by_country = sorted(minimal_retweets_by_country_dict.items(),
                                                key=lambda kv: kv[1], reverse=True)
    return sorted_minimal_retweets_by_country


def generate_retweets_plots(sorted_minimal_retweets_by_country):
    ''' Generates pie charts and bar graphs for retweets by country visualization '''
    labels = []
    slices = []

    for country_retweet_tup in sorted_minimal_retweets_by_country:
        labels.append(country_retweet_tup[0])
        slices.append(country_retweet_tup[1])

    # Plot the retweets by country bar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))

    wedges, texts, autotexts = plt.pie(slices[:5], autopct='%1.1f%%')
    plt.setp(autotexts, size=16, weight="bold")
    ax.legend(wedges, labels,
              title="Country",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              prop={'size': 16})
    ax.set_title("Number of #metoo Retweets by Country")
    plt.show()

    # For bar charts, we don't want the "unknown" label so discard it
    bar_labels = labels[0:2] + labels[3:8]
    bar_slices = slices[0:2] + slices[3:8]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Normalize the retweets
    bar_slices = [bar_slice / bar_slices[0]
                  for bar_slice in bar_slices]

    # Change United States to US and United Kingdom to UK
    bar_labels[0] = "US"
    bar_labels[2] = "UK"

    # Plot retweets by country bar chart
    ax.barh(bar_labels, bar_slices, align='center')
    ax.set_yticklabels(bar_labels, fontsize=16)
    plt.show()

    # From wikipedia
    english_speakers = [
        283160411,  # America
        125344737,  # India
        59600000,  # UK
        29973590,  # Canada
        5900000,  # Romania
        17357833,  # Australia
        28416,  # Kyrgyzstan
        50000000  # Pakistan
    ]

    # Normalize english_speakers to US
    english_speakers = [english_speaker / english_speakers[0]
                        for english_speaker in english_speakers]

    # Plot english speakers by country bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(bar_labels, english_speakers[:7], align='center')
    ax.set_yticklabels(bar_labels, fontsize=16)
    plt.show()
