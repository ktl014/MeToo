# MeToo
## Project to analyze spread and impact of the #MeToo movement on Twitter, over time and geography, different kinds of sentiments expressed etc.
### UCSD ECE 143 Spring '19 - Team 19
### Directory structure:
- datasets/ - datasets folder
  - .CSV file for tweets from 2017-18, 2018-19 and cumulative 2017-19 with details of tweets: text, date-time, username, #retweets, hashtags and mentions
  - file used to weed out emojis used in tweets for better sentiment analysis
- visualization_notebook.ipynb - sample plot of all the graphs used in the PPT. Time chart and mentions of famous people - part 1 of ppt - in 143_final_part 1 folder.
- freq_dict.py - contains functions:
  - freq_dict - return string:word mapping of important keywords/names found in tweets stored in CSVs
- get_tweet_with_more_stuffs.py - contains functions:
  - get_unique_tweets - save a CSV file containing tweet text, date, username, retweets, hashtags and mentions, given hashtag query string, language, since and until dates
- time_analysis_functions.py - contains functions:
  - csv_to_dataframe_time_amount - return dataframe object with month:#corresponding tweets mapping, given CSV datasets directory path
  - number_of_tweet_over_time - return bokeh object of #tweets over time given dataframe object of month:#tweets mapping
  - number_of_mention_per_person - return bokeh object of #mentions/persion in tweets given CSV dataset directory path
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
