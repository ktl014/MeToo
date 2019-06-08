import utilities as ut
from geopy.geocoders import Nominatim
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import math


def extract_retweet_numbers(df):
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
    # Create the tweepy api to extract username info (location)
    api = ut.create_api()

    # Create the geolocator to extract latitude and longitude from user location
    geolocator = Nominatim(user_agent="metwoo_data_analysis")

    # Read in dataframe
    df = pd.read_csv('datasets/tweet_2017_to_2018_with_hashtags.csv')

    # Extract locations from each tweet
    ut.extract_locations_from_file(api, geolocator, df, num_locs=100)


def get_dates_and_retweets():
    # Uncomment to extract locations from file (Otherwise, they will are in the repo)
    # extract_locations()
    # ut.generate_gif("images/global/")
    df = pd.read_csv('datasets/tweet_2017_to_2018_with_hashtags.csv')
    date_retweets_2017 = extract_retweet_numbers(df)

    df = pd.read_csv('datasets/tweet_2018_to_2019_with_hashtags.csv')
    date_retweets_2018 = extract_retweet_numbers(df)

    date_retweets_dict = {**date_retweets_2017, **date_retweets_2018}

    return date_retweets_dict


def format_lat_longs():
    lat_longs_2017 = pickle.load(open("lat_long_2017.p", "rb"))
    lat_longs_2018 = pickle.load(open("lat_long_2018_2019.p", "rb"))

    lat_longs_2017.reverse()
    lat_longs_2018.reverse()

    lat_longs_orig = lat_longs_2017 + lat_longs_2018

    lat_longs = ut.filter_lat_longs(lat_longs_orig)

    us_lat_longs = ut.filter_lat_longs(lat_longs_orig, us_only=True)

    return lat_longs, us_lat_longs


def get_retweets_by_country(lat_longs, date_retweets_dict):
    tweet_by_country_dict = {}
    retweets_by_country_dict = {}

    # Detetmine the country of each lat long
    for lat_long in lat_longs:
        country = ut.get_country(lon=lat_long[1][1], lat=lat_long[1][0])
        if country in tweet_by_country_dict:
            tweet_by_country_dict[country].append(lat_long)
        else:
            tweet_by_country_dict[country] = [lat_long]

    # For each country, determine the sum of retweets
    for country in tweet_by_country_dict:
        for tweet in tweet_by_country_dict[country]:
            retweets = ut.get_retweet_by_user_date(
                date_retweets_dict, tweet[0], tweet[2])
            if(country in retweets_by_country_dict):
                retweets_by_country_dict[country] += retweets
            else:
                retweets_by_country_dict[country] = retweets

    return retweets_by_country_dict


def get_minimal_retweets_by_country(retweets_by_country_dict):
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
    labels = []
    slices = []

    for country_retweet_tup in sorted_minimal_retweets_by_country:
        labels.append(country_retweet_tup[0])
        slices.append(country_retweet_tup[1])

    # Plot the retweets by country bar chart
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

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
    fig, ax = plt.subplots(figsize=(4, 6))

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


"""
ut.plot_locations(us_lat_longs, plot_us=True,
                  date_retweets_dict=date_retweets_dict, save_every=100, batch_size=100, color_gradient=False)
"""

"""
ut.plot_locations(lat_longs, plot_us=False,
                  date_retweets_dict=date_retweets_dict, save_every=100, batch_size=100, color_gradient=False)
"""

# ut.generate_gif("images/us/")

# ut.generate_gif("images/global/")
