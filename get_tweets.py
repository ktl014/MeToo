import tweepy
def create_api():
    consumer_key = "XXXX"
    consumer_secret = "XXXX"
    access_token = "XXXX"
    access_token_secret = "XXXX"
    auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)
    api = tweepy.API(auth)
    return api
api = create_api()
results = api.search(q="metoo",lang="en")
i = 1
for tweet in results:
    print (i,")))",tweet.user.screen_name,"Tweeted : ", tweet.text)
    i = i+1


