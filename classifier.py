from preprocess import preprocess, stop
from collections import defaultdict
from multiprocessing import Lock
from collections import Counter
from datetime import datetime
import pickle
import operator
import math


class Classifier(object):

    def __init__(self):
        # Data for Naive Bayes
        self.hc = {}
        self.htc = {}
        self.tc = {}
        self.thc = {}
        # Data for HF-IHU
        self.corpus = 0
        self.hfm = {}
        self.thfm = {}
        # Statistical
        self.tweet_total = 0
        self.hashtag_total = 0
        self.tweet_clean = 0
        self.one_hashtag = 0
        self.multi_hashtag = 0
        self.max_hashtag = 0
        self.non_hashtag = 0
        self.start_time = datetime.now()
        self.dump_time = datetime.now()
        # Mutex
        self.lock = Lock()
        # Read Backup
        self.state_load()

    def get_status(self):
        with self.lock:
            return self.tweet_total, self.tweet_clean, self.one_hashtag, self.multi_hashtag, self.non_hashtag, self.start_time, self.dump_time

    def state_load(self):
        with open('train_data.pickle', 'rb') as f:
            with self.lock:
                self.hc = pickle.load(f)
                self.tc = pickle.load(f)
                self.htc = pickle.load(f)
                self.thc = pickle.load(f)
                self.corpus = pickle.load(f)
                self.hfm = pickle.load(f)
                self.thfm = pickle.load(f)
                self.tweet_total = pickle.load(f)
                self.tweet_clean = pickle.load(f)
                self.non_hashtag = pickle.load(f)
                self.one_hashtag = pickle.load(f)
                self.multi_hashtag = pickle.load(f)
                self.max_hashtag = pickle.load(f)

                self.hashtag_total = 0
                for hashtag in self.hc.keys():
                    self.hashtag_total += self.hc[hashtag]

    def state_dump(self):
        with open('train_data.pickle', 'wb') as f:
            with self.lock:
                pickle.dump(self.hc, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.tc, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.htc, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.thc, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.corpus, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.hfm, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.thfm, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.tweet_total, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.tweet_clean, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.non_hashtag, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.one_hashtag, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.multi_hashtag, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.max_hashtag, f, pickle.HIGHEST_PROTOCOL)
                self.dump_time = datetime.now()

    def classify(self, method, tweet, results=None):
        score = defaultdict(int)
        if type(tweet) is list:
            term_input = [term.lower() for term in tweet]
        else:
            term_input = [term.lower() for term in tweet.split(" ")]
        if method == 'hfihu':
            for term in term_input:
                if term in self.thfm:
                    for hashtag in self.thfm[term]:
                        hf = self.thfm[term][hashtag] / self._get_hashtag_occurred(term)
                        ihu = math.log(self.corpus / self._get_term_occurred(hashtag))
                        score[hashtag] += hf * ihu
        elif method == 'nb':
            for hashtag in self._hashtags():
                score[hashtag] = self._prob(term_input, hashtag)

        sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
        return [hashtag[0] for hashtag in sorted_score[:results]]

    def train(self, tweet):

        count_hashtag = Counter()
        count_term = Counter()

        self.tweet_total += 1

        # Count hashtags only
        terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#') and len(term) > 1]
        count_hashtag.update(terms_hash)

        # Count terms only (no hashtags, no mentions)
        terms_only = [term for term in preprocess(tweet['text'], True) if
                      term.lower() not in stop and not term.startswith(('#', '@'))]
        count_term.update(terms_only)

        with self.lock:
            if len(terms_hash) == 1:
                self.one_hashtag += 1
            if len(terms_hash) >= 1:
                self.multi_hashtag += 1
            if len(terms_hash) > self.max_hashtag:
                self.max_hashtag = len(terms_hash)
            if len(terms_only) == 0:
                return
            if len(terms_hash) == 0:
                self.non_hashtag += 1
                return
            self.tweet_clean += 1
            self.corpus += len(terms_only)

            # Add term and hashtag for HF-IHU
            for hashtag in terms_hash:
                for term in terms_only:
                    if hashtag in self.hfm:
                        self.hfm[hashtag][term] = self.hfm[hashtag].get(term, 0) + count_term[term]
                    else:
                        temp = {term: count_term[term]}
                        self.hfm[hashtag] = temp
                    if term in self.thfm:
                        self.thfm[term][hashtag] = self.thfm[term].get(hashtag, 0) + count_hashtag[hashtag]
                    else:
                        temp = {hashtag: count_hashtag[hashtag]}
                        self.thfm[term] = temp

            # Add term and hashtag for Naive Bayes
            for term in terms_only:
                self.tc[term] = self.tc.get(term, 0) + count_term[term]
                if term in self.thc:
                    for hashtag in terms_hash:
                        self.thc[term][hashtag] = self.thc[term].get(hashtag, 0) + 1
                else:
                    self.thc[term] = dict.fromkeys(terms_hash, 1)

            for hashtag in terms_hash:
                self.hashtag_total += 1
                self.hc[hashtag] = self.hc.get(hashtag, 0) + count_hashtag[hashtag]
                if hashtag in self.htc:
                    for term in terms_only:
                        self.htc[hashtag][term] = self.htc[hashtag].get(term, 0) + 1
                else:
                    self.htc[hashtag] = dict.fromkeys(terms_only, 1)

    ################################################################################################
    # HF-IHU classification helper methods

    def _get_hashtag_occurred(self, term):
        sum = 0
        for hashtag in self.thfm[term]:
            sum += self.thfm[term][hashtag]
        return sum

    def _get_term_occurred(self, hashtag):
        sum = 0
        for term in self.hfm[hashtag]:
            sum += self.hfm[hashtag][term]
        return sum

    ################################################################################################
    # Naive Bayes classification helper methods

    def _htprob(self, terms, hashtag):
        p = 1
        for term in terms:
            p *= self._weightedprob(term, hashtag)
        return p

    def _fprob(self, term, hashtag):
        if term not in self.thc:
            return 0
        return self.thc[term].get(hashtag, 0) / self.hc[hashtag]

    def _weightedprob(self, term, hashtag, weight=1.0, ap=0.5):
        fprob = self._fprob(term, hashtag)
        totals = self.tc.get(term, 0)
        return ((weight * ap) + (totals * fprob)) / (weight + totals)

    def _hashtags(self):
        return self.hc.keys()

    def _hcount(self, hashtag):
        if hashtag in self.hc:
            return self.hc[hashtag]
        return 0

    def _totalcount(self):
        return self.hashtag_total

    def _prob(self, terms, hashtag):
        htprob = self._htprob(terms, hashtag)
        hprob = self._hcount(hashtag) / self._totalcount()
        return htprob * hprob
