import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time

import utils
import nn_utils

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="dmn_batch", help='embeding size (50, 100, 200, 300 only)')
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
args = parser.parse_args()

assert args.word_vector_size in [50, 100, 200, 300]

prefix = '%s.for%d.n%d%s.babi%s.adadelta' % (args.network, args.memory_hops, args.dim, 
    ".na" if args.normalize_attention else "", args.babi_id)

babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id)

word2vec = utils.load_glove(args.word_vector_size)

# init class
if args.network == 'dmn_batch':
    import dmn_batch
    dmn = dmn_batch.DMN_batch(babi_train_raw = babi_train_raw,
                                babi_test_raw = babi_test_raw,
                                word2vec = word2vec,
                                word_vector_size = args.word_vector_size,
                                dim = args.dim,
                                mode = args.mode,
                                answer_module = args.answer_module,
                                input_mask_mode = args.input_mask_mode,
                                memory_hops = args.memory_hops,
                                batch_size = args.batch_size,
                                l2 = args.l2,
                                normalize_attention = args.normalize_attention
                                )
elif args.network == 'dmn_basic':
    import dmn_basic
    if (args.batch_size != 1):
        raise Exception("no minibatch training, set batch_size=1")
    
    dmn = dmn_basic.DMN_basic(babi_train_raw = babi_train_raw,
                                babi_test_raw = babi_test_raw,
                                word2vec = word2vec,
                                word_vector_size = args.word_vector_size,
                                dim = args.dim,
                                mode = args.mode,
                                answer_module = args.answer_module,
                                input_mask_mode = args.input_mask_mode,
                                memory_hops = args.memory_hops,
                                l2 = args.l2,
                                normalize_attention = args.normalize_attention
                                )
 
elif args.network == 'dmn_qa':
    import dmn_qa
    if (args.batch_size != 1):
        raise Exception("no minibatch training, set batch_size=1")
    dmn = dmn_qa.DMN_qa(babi_train_raw = babi_train_raw,
                            babi_test_raw = babi_test_raw,
                            word2vec = word2vec,
                            word_vector_size = args.word_vector_size,
                            dim = args.dim,
                            mode = args.mode,
                            input_mask_mode = args.input_mask_mode,
                            memory_hops = args.memory_hops,
                            l2 = args.l2,
                            normalize_attention = args.normalize_attention
                            )

else: 
    raise Exception("No such network known: " + args.network)
    

if args.load_state != "":
    dmn.load_state(args.load_state) # of our class


def do_epoch(mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = step_data["skipped"]
        grad_norm = step_data["grad_norm"]
        param_norm = step_data["param_norm"]
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:
            avg_loss += current_loss
            
            for x in answers:
                y_true.append(x)
            
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
            
            # TODO: save the state sometimes
            print ("  %sing: %d.%03d / %d \t loss: %.3f \t avg_loss: %.3f \t pn: %.2f \t gn: %.2f \t skipped: %d \t %s" % 
                (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                 current_loss, avg_loss / (i + 1), param_norm, grad_norm, skipped, log))
            
        if np.isnan(param_norm):
            print "==> PARAM NORM IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t==p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
    
    return avg_loss, skipped


if args.mode == 'train':
    print "==> training"   	
    skipped = 0
    # TODO: shuffle!
    for epoch in range(args.epochs):
        start_time = time.time()
        _, skipped = do_epoch('train', epoch, skipped)
        
        epoch_loss, _ = do_epoch('test', epoch)
        
        state_name = 'states/%s.epoch%d.test%.5f.state' % (prefix, epoch, epoch_loss)
    
        print "==> saving ... %s" % state_name
        dmn.save_params(state_name, epoch)
        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

elif args.mode == 'test':
    do_epoch('test', 0)

else:
    raise Exception("unknown mode")