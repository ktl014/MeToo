### UCSD ECE 143 Spring '19 - Team 19

# \#MeToo hashtag analysis
## Project to analyze spread and impact of the #MeToo movement on Twitter, over time and geography, different kinds of sentiments expressed etc.

### Required Packages
#### The following packages can be installed through pip install -r requirements.txt:
* numpy
* pandas
* geopy
* matplotlib
* bokeh
* wordcloud - generates word clouds
* nltk - natural language toolkit
* tweepy - twitter api for accessing user data
* https://github.com/matplotlib/basemap/archive/master.zip - basemap for plotting geographical maps
* shapely - shape stuff 
* imageio - image manipulation/ saving
* pandas_bokeh
* GetOldTweets3 - extracts old tweets using urls

### Directory structure:
- datasets/ - datasets folder
  - .CSV file for tweets from 2017-18, 2018-19 and cumulative 2017-19 with details of tweets: text, date-time, username, #retweets, hashtags and mentions
  - file used to weed out emojis used in tweets for better sentiment analysis
- figs/ - plots folder - .PNG files of:
  - word clouds of proper and common nouns in tweets from 2017-18, 18-19 and cumulative
  - bar graph of occurrences of proper and common nouns in tweets from 2017-18, 18-19 and cumulative
  - bar graph of sentiments expressed in tweets from 2017-18
- images/ - map images folder - .PNG files of gif of temporal-expanding map in an ordered sequence for easy plotting
- processed_data/ - post-sentiment-analyzed-data - .CSV with same information as .CSV files from datasets/ folder and annotated with scores for different sentiments in additional columns
- visualization_notebook.ipynb - one sample of all types of graphs used in PPT
#### Remember to click external visualization icon on top right of notebook to render bokeh plots!
- \*.p files - pickle files containing lattitude and longitude information
- possible_names.txt
- requirements.txt - contains all dependencies required to run the project
- possible_names.txt - corpus containing 
- freq_dict.py - contains functions:
  - freq_dict - return string:word mapping of important keywords/names found in tweets stored in CSVs
- get_tweet_with_more_stuffs.py - contains functions:
  - get_unique_tweets - save a CSV file containing tweet text, date, username, retweets, hashtags and mentions, given hashtag query string, language, since and until dates (dependencies imported from https://github.com/Jefferson-Henrique/GetOldTweets-python)
- time_analysis_functions.py - contains functions:
  - csv_to_dataframe_time_amount - return dataframe object with month:#corresponding tweets mapping, given CSV datasets directory path
  - number_of_tweet_over_time - return bokeh object of #tweets over time given dataframe object of month:#tweets mapping
  - number_of_mention_per_person - return bokeh object of #mentions/persion in tweets given CSV dataset directory path
- tweet_text_analysis.py - contains functions:
  - clean_tweet - strips tweet text of punctuations, special characters, hyperlinks, NLTK stopwords, PorterStem of words
  - remove_emoji - removes emoji from tweet texts
  - couroutines - for tracing progress of computing sentiment - including percentage metrics
  - plot_sentiment - plots bar graph of sentiment of tweets, based on sentiment corpus in dataset folder
  - class NRCLexicon - NRC Lexicon to parse, map and score tweets, functions:
    - init
    - lemnatize - return comma-separated lemnatized words given query string
    - sentiment score - return sentiment score given list of words
- utilities.py - contains functions:
  - create_country_dict - return dictionary of countries based on geojson file from githubusercontent
  - get_country - return country given longitude,lattitude tuple
  - get_retweet_by_user_date - return number of retweets of a tweet given dictionary of tweets, username and date of required tweet
  - get_user_lat_long - return longitude,lattitude tuple of user's location given tweepy API object, geolocator object and username
  - extract_locations_from_file - saves all usernames and lattitude,longitude info to pickle file
  - create_api - creates API object to scrape tweets from twitter with tweepy module
  - plot_locations - plot map to geographically visualize #metoo tweets with scaling bubble based on retweet count
  - filter_lat_longs - return list of lattitudes and longitudes from USA alone
  - generate_gif - generate gif from files in given directory (containing stream of images)
  - extract_retweet_numbers - return dictionary of date:#retweets mapping given dataframe of full csv file
  - extract_locations - calls extract_locations_from_file and passes created api, geolocator object, dataframe of dataset csv and number of locations)
  - get_dates_and_retweets - return dictionary of combined dates and associated retweets from csv files
  - format_lat_longs - reverse and combines lats and longs and returns only USA locations
  - get_retweets_by_country - return country-wise retweets dictionary
  - get_minimal_retweets_by_country - return subset of retweets_by_country based on retweet_cnt
  - generate_retweets_plots - gnenerate pie charts and bar graphs for retweets by country
