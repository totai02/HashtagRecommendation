from preprocess import preprocess, stop
from multiprocessing import Lock
from datetime import datetime
import pickle


class Evaluate(object):

    def __init__(self, classifier):
        self.number_of_rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        self.hfihu_sumHS = {}
        self.hfihu_sumSi = {}
        self.hfihu_sumHi = {}

        self.nb_sumHS = {}
        self.nb_sumSi = {}
        self.nb_sumHi = {}

        self.tweets_train = 0
        self.tweets_test = 0
        self.dump_time = 0

        self.classifier = classifier
        self.tweets = []

        self.lock = Lock()

        self.state_load()

    def get_status(self):
        hfihu_precision = {}
        hfihu_recall = {}
        nb_precision = {}
        nb_recall = {}
        for num in self.number_of_rank:
            if num <= 10:
                hfihu_precision[num] = self.hfihu_sumHS[num] / self.hfihu_sumSi[num]
                nb_precision[num] = self.nb_sumHS[num] / self.nb_sumSi[num]
            if num == 1 or num >= 10:
                hfihu_recall[num] = self.hfihu_sumHS[num] / self.hfihu_sumHi[num]
                nb_recall[num] = self.nb_sumHS[num] / self.nb_sumHi[num]
        return hfihu_precision, hfihu_recall, nb_precision, nb_recall, self.tweets_train, self.tweets_test, self.dump_time

    def add_tweet(self, tweet):
        terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#') and len(term) > 1]
        terms_only = [term for term in preprocess(tweet['text'], True) if
                      term.lower() not in stop and not term.startswith(('#', '@'))]
        if len(terms_only) == 0 or len(terms_hash) == 0:
            return False
        with self.lock:
            tweet = {"terms": terms_only, "hashtags": terms_hash}
            self.tweets.append(tweet)
        return True

    def evaluate(self):
        with self.lock:
            for tweet in self.tweets:
                hfihu_hashtags = self.classifier.classify("hfihu", tweet['terms'], 100)
                nb_hashtags = self.classifier.classify("nb", tweet['terms'], 100)

                for num in self.number_of_rank:
                    for si in hfihu_hashtags[:num]:
                        if si in tweet['hashtags']:
                            self.hfihu_sumHS[num] = self.hfihu_sumHS.get(num, 0) + 1
                    self.hfihu_sumHi[num] = self.hfihu_sumHi.get(num, 0) + len(tweet['hashtags'])
                    self.hfihu_sumSi[num] = self.hfihu_sumSi.get(num, 0) + num

                for num in self.number_of_rank:
                    for si in nb_hashtags[:num]:
                        if si in tweet['hashtags']:
                            self.nb_sumHS[num] = self.nb_sumHS.get(num, 0) + 1
                    self.nb_sumHi[num] = self.nb_sumHi.get(num, 0) + len(tweet['hashtags'])
                    self.nb_sumSi[num] = self.nb_sumSi.get(num, 0) + num

            self.tweets_train = self.classifier.tweet_clean
            self.tweets_test = len(self.tweets)
            self.state_dump()
            self.tweets.clear()

    def state_load(self):
        with open("evaluate.pickle", "rb") as f:
            with self.lock:
                self.hfihu_sumHi = pickle.load(f)
                self.hfihu_sumSi = pickle.load(f)
                self.hfihu_sumHS = pickle.load(f)
                self.nb_sumHi = pickle.load(f)
                self.nb_sumSi = pickle.load(f)
                self.nb_sumHS = pickle.load(f)
                self.tweets_train = pickle.load(f)
                self.tweets_test = pickle.load(f)
                self.dump_time = pickle.load(f)

    def state_dump(self):
        with open("evaluate.pickle", "rb") as f:
            with self.lock:
                self.dump_time = datetime.now()
                pickle.dump(self.hfihu_sumHi, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.hfihu_sumSi, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.hfihu_sumHS, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.nb_sumHi, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.nb_sumSi, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.nb_sumHS, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.tweets_train, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.tweets_test, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.dump_time, f, pickle.HIGHEST_PROTOCOL)
