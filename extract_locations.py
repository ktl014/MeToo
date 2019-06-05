import utilities as ut
from geopy.geocoders import Nominatim
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm

# Create the tweepy api to extract username info (location)
api = ut.create_api()

# Create the geolocator to extract latitude and longitude from user location
geolocator = Nominatim(user_agent="metwoo_data_analysis")

# Read in dataframe
df = pd.read_csv('datasets/tweet_2017_to_2018_with_hashtags.csv')

# Extract locations from each tweet
ut.extract_locations_from_file(api, geolocator, df)

lat_longs = pickle.load(open("lat_long_2017.p", "rb"))

lats = []
longs = []
for lat_long in lat_longs:
    latitude = lat_long[1][0]
    longitude = lat_long[1][1]
    if (latitude is not None and longitude is not None):
        lats.append(latitude)
        longs.append(longitude)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Create map of US
m_us = Basemap(projection="lcc", lat_1=33, lon_0=-95,
               llcrnrlat=22, urcrnrlat=49,
               llcrnrlon=-119, urcrnrlon=-64)
# Create map of World
m_world = Basemap()

m_us.drawcoastlines()
m_us.drawstates()
m_us.drawcountries()

# Plot the locations
m_us.scatter(longs, lats, latlon=True, marker='o')

plt.show()
