import sys
import os
import json

loaded_models = {}

current_dir = os.getcwd()


def load_network(code_path, babi_train_raw, babi_test_raw, word2vec, word_vector_size, model_json):
    if model_json in loaded_models:
        print "!> model %s is already loaded" % model_json
        return loaded_models[model_json]

    print "!> loading model %s..." % model_json

    model_file = open(model_json)
    args_dict = json.load(model_file)

    assert word_vector_size == args_dict['word_vector_size']

    args_dict['babi_train_raw'] = babi_train_raw
    args_dict['babi_test_raw'] = babi_test_raw
    args_dict['word2vec'] = word2vec

    # init class
    if args_dict['network'] == 'dmn_batch':
        raise Exception("dmn_batch did not implement predict()")
        sys.path.insert(0, code_path)
        import dmn_batch
        dmn = dmn_batch.DMN_batch(**args_dict)
        sys.path.insert(0, current_dir)


    elif args_dict['network'] == 'dmn_basic':
        raise Exception("dmn_batch did not implement predict()")
        sys.path.insert(0, code_path)
        import dmn_basic
        dmn = dmn_basic.DMN_basic(**args_dict)
        sys.path.insert(0, current_dir)


    elif args_dict['network'] == 'dmn_smooth':
        sys.path.insert(0, code_path)
        import dmn_smooth
        dmn = dmn_smooth.DMN_smooth(**args_dict)
        sys.path.insert(0, current_dir)

    elif args_dict['network'] == 'dmn_qa':
        raise Exception("dmn_batch did not implement predict()")

        sys.path.insert(0, code_path)
        import dmn_qa_draft
        dmn = dmn_qa_draft.DMN_qa(**args_dict)
        sys.path.insert(0, current_dir)

    else:
        raise Exception("No such network known: " + args_dict['network'])

    print "!> loading state %s..." % args_dict['load_state']

    dmn.load_state(args_dict['load_state'])

    loaded_models[model_json] = dmn

    return dmn


def predict(code_path, babi_train_raw, babi_test_raw, word2vec, word_vector_size, model_json, context, question):
    dmn = load_network(code_path, babi_train_raw, babi_test_raw, word2vec, word_vector_size, model_json)
    data = [{
        "Q": question,
        "C": context
    }]
    probabilities, attentions = dmn.predict(data)

    return {
        'answer': dmn.ivocab[probabilities.argmax()],
        'confidence': str(probabilities.max()),
        'episodes': attentions.tolist(),
        'facts': context.split(' . ')
    }
