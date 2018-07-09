from classifier_manager import classifier, evaluate, listener, train_classifier, dump_classifier, dump_evaluate, \
    Interval
from flask_restful import Resource, Api, reqparse
from flask import Flask, render_template, request
from flask_bower import Bower
from threading import Thread
import requests
import json

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer

app = Flask(__name__)
Bower(app)

api = Api(app)


@app.route("/")
def index():
    return render_template('tweet.html', method="hfihu")


@app.route("/classifier/hfihu")
def hfihu_classifier():
    return render_template('tweet.html', method="hfihu")


@app.route("/classifier/nb")
def nb_classifier():
    return render_template('tweet.html', method="nb")


@app.route("/stats/classifier")
def stats_classifier():
    results = requests.get(request.url_root + "api/status/classifier")
    results.raise_for_status()
    return results.text


@app.route("/stats/evaluate")
def stats_evaluate():
    return ""


class ClassificationAPI(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('text', type=str, required=True, help='No text to classify was provided',
                                   location='json')
        self.reqparse.add_argument('results', type=int, required=False, default=5, location='json')
        super(ClassificationAPI, self).__init__()

    def post(self, method):
        args = self.reqparse.parse_args()
        return classifier.classify(method, args['text'], results=args['results']), 201


api.add_resource(ClassificationAPI, "/api/classify/<string:method>")


class ClassifierStatusAPI(Resource):
    def get(self):
        tweet_total, tweet_clean, one_hashtag, multi_hashtag, non_hashtag, start_time, dump_time = classifier.get_status()
        return {
            'training': {
                'tweet_total': tweet_total,
                'tweet_clean': tweet_clean,
                'one_hashtag': one_hashtag,
                'multi_hashtag': multi_hashtag,
                'non_hashtag': non_hashtag,
            },
            'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'dump_time': dump_time.strftime("%Y-%m-%d %H:%M:%S")
        }


class EvaluateStatusAPI(Resource):
    def get(self):
        hfihu_precision, hfihu_recall, nb_precision, nb_recall, tweets_train, tweets_test, dump_time = evaluate.get_status()
        return {
            'hfihu': {
                'precision': hfihu_precision,
                'recall': hfihu_recall,
            },
            'nb': {
                'precision': nb_precision,
                'recall': nb_recall,
            },
            'tweets_train': tweets_train,
            'tweets_test': tweets_test,
            'dump_time': dump_time.strftime("%Y-%m-%d %H:%M:%S")
        }


class DownloaderStatusAPI(Resource):
    def get(self):
        total_tweet, start_time = listener.get_status()
        return {
            "total_tweet": total_tweet,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S")
        }


api.add_resource(ClassifierStatusAPI, '/api/status/classifier')
api.add_resource(EvaluateStatusAPI, '/api/status/evaluate')
api.add_resource(DownloaderStatusAPI, '/api/status/downloader')


def dev_server():
    trainer = Thread(target=train_classifier)
    trainer.start()

    sched_classifier = Interval(3600, dump_classifier)
    sched_classifier.start()

    sched_evaluate = Interval(2 * 3600, dump_evaluate)
    sched_evaluate.start()

    app.run(debug=True)

    trainer.join()
    sched_classifier.join()
    sched_evaluate.join()


def server():
    trainer = Thread(target=train_classifier)
    trainer.start()

    sched_classifier = Interval(3600, dump_classifier)
    sched_classifier.start()

    sched_evaluate = Interval(2 * 3600, dump_evaluate)
    sched_evaluate.start()

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)
    IOLoop.instance().start()

    trainer.join()
    sched_classifier.join()
    sched_evaluate.join()


if __name__ == '__main__':
    dev_server()
