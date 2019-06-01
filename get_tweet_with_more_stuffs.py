import GetOldTweets3 as got

tweetCriteria = got.manager.TweetCriteria().setQuerySearch('#metoo').setTopTweets(True).setLang("en").setSince('2019-05-01').setUntil('2019-05-21')

tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]


import csv
with open('tweet_text_date.csv', 'w', newline='',encoding='utf-8') as csvfile:
    fieldnames = ['text', 'date','user','retweets','mentions']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for tweet in got.manager.TweetManager.getTweets(tweetCriteria):

        if tweet.retweets > 10 and hasattr(tweet, 'username'):
            writer.writerow({'text': tweet.text, 'date': tweet.date,'user':tweet.username, 'retweets': tweet.retweets,'mentions':tweet.mentions })

print("done")