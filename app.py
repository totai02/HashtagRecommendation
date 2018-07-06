from flask import Flask, render_template
from flask_bower import Bower
from flask_restful import Resource, Api, reqparse
from apscheduler.scheduler import Scheduler
from multiprocessing import Process
from classifier_manager import classifier, evaluate, train_classifier, dump_classifier, dump_evaluate

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
    return ""


@app.route("/stats/evaluate")
def stats_classifier():
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


class StatusClassifierAPI(Resource):
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


class StatusEvaluateAPI(Resource):
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


api.add_resource(StatusClassifierAPI, '/api/classifier/status')
api.add_resource(StatusEvaluateAPI, '/api/evaluate/status')


def dev_server():
    trainer = Process(target=train_classifier)
    trainer.start()

    sched_classifier = Scheduler()
    sched_classifier.start()
    sched_classifier.add_interval_job(dump_classifier, hours=1)

    sched_evaluate = Scheduler()
    sched_evaluate.start()
    sched_evaluate.add_interval_job(dump_evaluate, hours=2)

    app.run(debug=True)


def server():
    trainer = Process(target=train_classifier)
    trainer.start()

    sched_classifier = Scheduler()
    sched_classifier.start()
    sched_classifier.add_interval_job(dump_classifier, hours=1)

    sched_evaluate = Scheduler()
    sched_evaluate.start()
    sched_evaluate.add_interval_job(dump_evaluate, hours=2)

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)
    IOLoop.instance().start()


if __name__ == '__main__':
    dev_server()
