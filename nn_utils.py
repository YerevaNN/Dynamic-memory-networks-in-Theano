import theano
import theano.tensor as T

def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out
    
def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])