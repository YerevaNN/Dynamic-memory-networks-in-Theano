import codecs
import os
import json

import re


def read_networks(base_path):
    networks_path = os.path.join(base_path, 'networks')
    networks = [d for d in os.listdir(networks_path) if os.path.isdir(os.path.join(networks_path, d))]

    networks_info = []

    for network in networks:
        with codecs.open(os.path.join(networks_path, network, 'info.json'), 'r', 'utf-8') as f:
            networks_info.append(json.load(f))

    return networks_info


def read_models(base_path, network):
    models_path = os.path.join(base_path, 'networks', network, 'models')
    models = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f))]

    models_info = []

    for model in models:
        with codecs.open(os.path.join(models_path, model), 'r', 'utf-8') as f:
            model_info = json.load(f)
            models_info.append(
                    {'id': model_info['id'], 'name': model_info['name'], 'description': model_info['description']})

    return models_info


def read_model(base_path, network, model):
    model_path = os.path.join(base_path, 'networks', network, 'models', model + '.json')

    with codecs.open(model_path, 'r', 'utf-8') as f:
        return json.load(f)


punction_re = re.compile(r"(\b|[.?!]+)$", re.MULTILINE)


def babify(story, question):
    context = ' '.join([punction_re.sub(' .', fact.strip()) for fact in story.split('\n')])
    question = punction_re.sub('', question.strip())

    return context, question


def get_model_json_path(base_path, network, model):
    return os.path.join(base_path, 'networks', network, 'models', model + '.json')
