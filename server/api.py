import imp
import dmn_helper
import os
import utils
import argparse
from flask import Flask
from flask_restful import reqparse, Api, Resource

app = Flask(__name__, static_folder='ui', static_url_path='')
api = Api(app)

api_prefix = '/api/v1'

code_path = data_path = None
word_vector_size = word2vec = babi_train_raw = babi_test_raw = None

request_parser = reqparse.RequestParser()
request_parser.add_argument('question', type=str, required=True, help='Question of the task')
request_parser.add_argument('story', type=str, required=True, help='Story of the task')

@app.route('/')
def root():
    return app.send_static_file('index.html')

class NetworksList(Resource):
    def get(self):
        return utils.read_networks(data_path)


class Network(Resource):
    def get(self):
        return utils.read_networks(data_path)


class ModelsList(Resource):
    def get(self, network):
        return utils.read_models(data_path, network)


class Model(Resource):
    def get(self, network, model):
        return utils.read_model(data_path, network, model)


class Predict(Resource):
    def post(self, network, model):
        predict_args = request_parser.parse_args()

        context, question = utils.babify(predict_args['story'], predict_args['question'])
        model_json = utils.get_model_json_path(data_path, network, model)

        return dmn_helper.predict(code_path, babi_train_raw, babi_test_raw, word2vec, word_vector_size, model_json,
                                  context,
                                  question)


api.add_resource(NetworksList, api_prefix + '/networks', endpoint='networks')
api.add_resource(ModelsList, api_prefix + '/networks/<string:network>/models', endpoint='models_by_networks')
api.add_resource(Model, api_prefix + '/networks/<string:network>/models/<string:model>', endpoint='model')
api.add_resource(Predict, api_prefix + '/networks/<string:network>/models/<string:model>/_predict', endpoint='predict')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=str, default="..", help='folder containing network\'s code')
    parser.add_argument('--data', type=str, default="./data", help='folder containing networks, models, etc.')
    parser.add_argument('--debug', type=int, default=0, help='run server in debug mode')
    parser.add_argument('--word_vector_size', type=int, default=50,
                        help='embeding size (50, 100, 200, 300 only)')  # let's fix to 50

    args = parser.parse_args()

    code_path = args.code
    data_path = args.data

    dmn_utils = imp.load_source('utils', os.path.join(code_path, 'utils.py'))
    babi_train_raw, babi_test_raw = dmn_utils.get_babi_raw('joint', 'joint')
    word2vec = dmn_utils.load_glove(args.word_vector_size)

    word_vector_size = args.word_vector_size

    print "==> running server"

    app.run(host='0.0.0.0', debug=args.debug == 1)
