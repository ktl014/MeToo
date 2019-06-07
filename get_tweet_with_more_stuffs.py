def get_unique_tweets(hashtag = '#metoo', lang = "en",since ='2018-10-17', until = '2019-05-21'):
    """get tweets from tweeter with given parameters
    hashtage: hashtage
    lang: language of the tweet
    since: time fromat in yyyy-mm-rr
    until: time fromat in yyyy-mm-rr"""
    assert(isinstance(hashtag,str) and isinstance(lang,str) and isinstance(since,str) and isinstance(until,str))

    for i in since.split('-'):
        assert(i.isdigit())
    for i in until.split('-'):
        assert(i.isdigit())

    import GetOldTweets3 as got
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(hashtag).setTopTweets(True).setLang(lang).setSince(since).setUntil(until)

    tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]

    import csv
    with open('tweet_%s_to_%s_with_hashtags.csv' %(since[:4],until[:4]), 'w', newline='',encoding='utf-8') as csvfile:
        fieldnames = ['text', 'date','user','retweets','hashtags','mentions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for tweet in got.manager.TweetManager.getTweets(tweetCriteria):

            if tweet.retweets > 0 and hasattr(tweet, 'username'):
                writer.writerow({'text': tweet.text, 'date': tweet.date,'user':tweet.username, 'retweets': tweet.retweets,'hashtags': tweet.hashtags,'mentions':tweet.mentions })

    print("done")
    return 0