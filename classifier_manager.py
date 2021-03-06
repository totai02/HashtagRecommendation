from tweepy.streaming import StreamListener
import threading
from threading import Timer
from classifier import Classifier
from evaluate import Evaluate
from datetime import datetime
from tweepy import OAuthHandler
from tweepy import Stream
import json


class Interval:

    def __init__(self, t, func):
        self.t = t
        self.func = func
        self.thread = Timer(self.t, self.handle_function)

    def handle_function(self):
        self.func()
        self.thread = Timer(self.t, self.handle_function)
        self.thread.start()

    def start(self):
        self.thread.start()

    def cancel(self):
        self.thread.cancel()

    def join(self):
        self.thread.join()


class MyListener(StreamListener):

    def __init__(self):
        self.max_tweets_per_second = 10
        self.current_tweets = 0
        self.start_time = datetime.now()
        self.last_time = datetime.now()
        self.count_tweet = 0
        self.total_tweet = 0
        super(MyListener, self).__init__()

    def get_status(self):
        return self.total_tweet, self.start_time

    def on_data(self, data):
        tweet = json.loads(data)
        if 'text' in tweet:
            if "RT @" in tweet['text'] or tweet['retweeted'] is True:
                return True
        else:
            return True

        self.total_tweet += 1

        if (datetime.now() - self.last_time).seconds > 0:
            self.current_tweets = 0
            self.last_time = datetime.now()
        else:
            if self.current_tweets >= self.max_tweets_per_second:
                return True

        self.count_tweet += 1
        if self.count_tweet > 9:
            if evaluate.add_tweet(tweet):
                self.count_tweet = 0
        else:
            classifier.train(tweet)

        # Backup
        # try:
        #     with open('worldcup.json', 'a') as f:
        #         self.current_tweets += 1
        #         f.write(data)
        #         return True
        # except BaseException as e:
        #     print("Error on_data: %s" % str(e))
        # return True

    def on_error(self, status):
        print(status)
        return True


classifier = Classifier()
evaluate = Evaluate(classifier)
listener = MyListener()


def train_classifier():
    consumer_key = 'JLZAWxT74QZ4gFBhZvW1G2WUd'
    consumer_secret = 'W8qQPm82bOtJy744rZuJ52JhNsrMHzCnjXU54UEpG9oFJTtr96'
    access_token = '3236157257-CEbv8yEVjPBL4g6IZJDPAMwotsROQgTXFQoTfcF'
    access_secret = '2l0F0UEyw9BCPGdODSFuV6Gx84mAEdE3nytx6WULusmx1'

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, listener)
    filtro = ['worldcup', '#worldcup', 'world cup']
    while True:
        try:
            twitter_stream.filter(track=filtro, languages=['en'])
        except BaseException as e:
            print("Error on streaming: %s" % str(e))


def dump_classifier():
    classifier.state_dump()
    print("Dump classifier: " + str(datetime.now()))


def dump_evaluate():
    evaluate.state_dump()
    print("Dump evaluate: " + str(datetime.now()))


