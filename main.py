import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json

import utils
import nn_utils

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="dmn_batch", help='network type: dmn_basic, dmn_smooth, or dmn_batch')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=5, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--babi_id', type=str, default="1", help='babi task ID')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--babi_test_id', type=str, default="", help='babi_id of test set (leave empty to use --babi_id)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.set_defaults(shuffle=True)
args = parser.parse_args()

print args

assert args.word_vector_size in [50, 100, 200, 300]

network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.babi%s' % (
    args.network, 
    args.memory_hops, 
    args.dim, 
    args.batch_size, 
    ".na" if args.normalize_attention else "", 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "",
    args.babi_id)


babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_test_id)

word2vec = utils.load_glove(args.word_vector_size)

args_dict = dict(args._get_kwargs())
args_dict['babi_train_raw'] = babi_train_raw
args_dict['babi_test_raw'] = babi_test_raw
args_dict['word2vec'] = word2vec
    

# init class
if args.network == 'dmn_batch':
    import dmn_batch
    dmn = dmn_batch.DMN_batch(**args_dict)

elif args.network == 'dmn_basic':
    import dmn_basic
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_basic.DMN_basic(**args_dict)

elif args.network == 'dmn_smooth':
    import dmn_smooth
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_smooth.DMN_smooth(**args_dict)

elif args.network == 'dmn_qa':
    import dmn_qa_draft
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_qa_draft.DMN_qa(**args_dict)

else: 
    raise Exception("No such network known: " + args.network)
    

if args.load_state != "":
    dmn.load_state(args.load_state)


def do_epoch(mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:
            avg_loss += current_loss
            
            for x in answers:
                y_true.append(x)
            
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
    
    return avg_loss, skipped


if args.mode == 'train':
    print "==> training"   	
    skipped = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if args.shuffle:
            dmn.shuffle_train_set()
        
        _, skipped = do_epoch('train', epoch, skipped)
        
        epoch_loss, skipped = do_epoch('test', epoch, skipped)
        
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)

        if (epoch % args.save_every == 0):    
            print "==> saving ... %s" % state_name
            dmn.save_params(state_name, epoch)
        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    do_epoch('test', 0)

else:
    raise Exception("unknown mode")
