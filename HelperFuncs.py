from collections import OrderedDict
import numpy as np
import theano as theano
import theano.tensor as T
from theano.ifelse import ifelse


######################
# PARAM UPDATE FUNCS #
######################

def norm_clip(dW, max_l2_norm=10.0):
    """
    Clip theano symbolic var dW to have some max l2 norm.
    """
    dW_l2_norm = T.sqrt(T.sum(dW**2.0))
    norm_ratio = (max_l2_norm / dW_l2_norm)
    clip_factor = ifelse(T.lt(norm_ratio, 1.0), norm_ratio, 1.0)
    dW_clipped = dW * clip_factor
    return dW_clipped

def get_param_updates(params=None, grads=None, \
        alpha=None, beta1=None, beta2=None, it_count=None, \
        mom2_init=1e-3, smoothing=1e-6, max_grad_norm=10000.0):
    """
    This update has some extra inputs that aren't used. This is just so it
    can be called interchangeably with "ADAM" updates.
    """

    # make an OrderedDict to hold the updates
    updates = OrderedDict()
    # alpha is a shared array containing the desired learning rate
    lr_t = alpha[0]
    
    for p in params:
        # get gradient for parameter p
        grad_p = norm_clip(grads[p], max_grad_norm)

        # initialize first-order momentum accumulator
        mom1_ary = 0.0 * p.get_value(borrow=False)
        mom1 = theano.shared(mom1_ary)
        
        # update momentum accumulator
        mom1_new = (beta1[0] * mom1) + ((1. - beta1[0]) * grad_p)
        
        # do update
        p_new = p - (lr_t * mom1_new)

        # apply updates to 
        updates[p] = p_new
        updates[mom1] = mom1_new
    return updates