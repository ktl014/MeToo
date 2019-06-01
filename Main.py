from nameparser.parser import HumanName
import sys
import get_tweets
import gmplot
from geopy.geocoders import Nominatim
import pickle
import nltk
# nltk.download()

if sys.version_info[0] < 3:
    import got
else:
    import got3 as got


def get_human_names(text):
    # From: https://stackoverflow.com/questions/20290870/improving-the-extraction-of-human-names-with-nltk
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary=False)
    person_list = []
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1:  # avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    return (person_list)


def get_common_names_from_file():
    f = open("possible_names.txt", "rb")

    possible_names = f.readlines()

    common_name_dict = {}

    for string in possible_names:
        string = string.rstrip()
        for name in get_human_names(str(string)):
            if (name in common_name_dict):
                common_name_dict[name] += 1
            else:
                common_name_dict[name] = 1

    # print(common_name_dict)

    for k, v in sorted(common_name_dict.items(), key=lambda kv: kv[1]):
        print(k, v)

    return common_name_dict


def get_user_lat_long(api, geolocator, username):
    ''' Takes in a tweepy api, a geolocator object and a twitter username and
    returns the (latitude, longitude) tuple of the user's location if it 
    exists. If not, (None, None) is returned '''

    longitude = None
    latitude = None
    user = api.get_user(username)

    if (user is not None):
        raw_location = user.location
        location = geolocator.geocode(raw_location)

        if (location is not None):
            longitude = location.longitude
            latitude = location.latitude

    return (latitude, longitude)


def main():
    # To regenerate and save common names stripped from file
    # name_dict = get_common_names_from_file()
    # pickle.dump(name_dict, open("name_dict.p", "wb"))

    # To load common names dict from pickle file
    name_dict = pickle.load(open("name_dict.p", "rb"))

    # For storing name counts found in tweets
    tweet_names_dict = {}

    # Create the tweepy api to extract username info (location)
    api = get_tweets.create_api()

    geolocator = Nominatim(user_agent="metwoo_data_analysis")

    def printTweet(descr, t):
        print(descr)
        print("Username: %s" % t.username)
        print("Retweets: %d" % t.retweets)
        print("Text: %s" % t.text)
        print("Mentions: %s" % t.mentions)
        print("Hashtags: %s\n" % t.hashtags)

        # Get the location of the user
        (latitude, longitude) = get_user_lat_long(api, geolocator, t.username)
        if(latitude is None or longitude is None):
            print("No location data found for user")
        else:
            print("Long:", longitude, " Lat: ", latitude)

        # Extract people names from the tweet
        names = get_human_names(t.text)
        associated_names = []
        # Go through each name in the tweet and see if it's one of the names
        # we've scraped of possible names involved in the metoo movement
        for name in names:
            if name in name_dict:
                if(name in tweet_names_dict):
                    tweet_names_dict[name] += 1
                else:
                    tweet_names_dict[name] = 1
                associated_names.append(name)

        # Note: A lot of tweets will have an empty list for this.
        # TODO: also search for twitter handle usernames that are @ed and their
        # real name to see if they are in the tweet_names_dict
        print("Associated names: ", associated_names)

    # Example 1 - Get tweets by username
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(
        '#metoo').setMaxTweets(100)

    tweets = got.manager.TweetManager.getTweets(tweetCriteria)

    for tweet in tweets:
        printTweet("", tweet)

    print(tweet_names_dict)


if __name__ == '__main__':
    main()
