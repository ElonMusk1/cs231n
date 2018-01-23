from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        H   = input_dim[1]
        HH  = filter_size
        pad = (filter_size - 1) // 2
        stride_conv = 1
        stride_pool = 2
        H_pool = 2
        conv_size = 1 + (H + 2 * pad - HH) / stride_conv
        pool_size = 1 + (conv_size - H_pool) / stride_pool
        w2_size = int(num_filters*pool_size*pool_size)
       
        W1 = weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
        W2 = weight_scale*np.random.randn(w2_size,hidden_dim)
        W3 = weight_scale*np.random.randn(hidden_dim,num_classes)
        b1 = np.zeros(num_filters)
        b2 = np.zeros(hidden_dim)
        b3 = np.zeros(num_classes)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out = []
        cache = []
        
        # conv
        out_conv, cache_conv = conv_forward_naive(X, W1, b1, conv_param)
        
        # relu
        out_relu, cache_relu = relu_forward(out_conv)
        
        # max pool
        out_pool, cache_pool = max_pool_forward_naive(out_relu, pool_param)
                
        # af-relu
        out_afrel, cache_afrel = affine_relu_forward(out_pool, W2, b2)
        
        # last affine
        scores, cache_last = affine_forward(out_afrel, W3, b3)
        
        out   = [out_conv,   out_relu,   out_pool,   out_afrel,   scores]
        cache = [cache_conv, cache_relu, cache_pool, cache_afrel, cache_last]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reg = self.reg
        
        loss, dscores = softmax_loss(scores, y)
        loss_reg = 0.5*reg * np.sum(W3 * W3) + 0.5*reg * np.sum(W2 * W2) + 0.5*reg * np.sum(W1 * W1)
        loss += loss_reg
        
        # last affine
        dout_afrel, dW3, db3 = affine_backward(dscores, cache[4])
        
        # af-relu
        dout_pool, dW2, db2 = affine_relu_backward(dout_afrel, cache[3])
        
        # max pool
        dout_relu = max_pool_backward_naive(dout_pool, cache[2])
        
        # relu
        dout_conv = relu_backward(dout_relu, cache[1])
        
        # conv
        dx, dW1, db1 = conv_backward_naive(dout_conv, cache[0])
        
        grads['W3'] = dW3
        grads['b3'] = db3
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        # add regularization gradient contribution
        grads['W3'] += reg * W3
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
