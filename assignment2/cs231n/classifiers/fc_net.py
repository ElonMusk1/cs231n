from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
     
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        W1 = weight_scale*np.random.randn(input_dim,hidden_dim)
        W2 = weight_scale*np.random.randn(hidden_dim,num_classes)
        b1 = np.zeros(hidden_dim)
        b2 = np.zeros(num_classes)
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        # Unhttp://localhost:8888/edit/cs231n/classifiers/fc_net.py#pack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg
        num_train = X.shape[0]
        N = X.shape[0]
        D = np.prod([d for d in X.shape[1:]]) 
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        x_re = X.reshape(N,D) 
        
        # ручная реализация 
        #hidden_layer = np.maximum(0, x_re.dot(W1) + b1) # ReLU activation  
        #scores = hidden_layer.dot(W2) + b2
        
        out1, cache1 = affine_relu_forward(x_re, W1, b1)
        scores, cache2 = affine_forward(out1, W2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        '''
        # ручная реализация
        scores -= np.max(scores, axis=1, keepdims=True)  #to avoid numerical instability
        scores_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = np.exp(scores)/scores_exp
        loss = np.sum(-np.log(softmax[np.arange(num_train), y]))
        loss /= num_train
        loss_reg = 0.5*reg * np.sum(W2 * W2) + 0.5*reg * np.sum(W1 * W1)
        loss += loss_reg
        
        # compute the gradient on scores
        dscores = softmax
        dscores[range(num_train),y] -= 1
        dscores /= num_train
        # W2 and b2
        grads['W2'] = np.dot(hidden_layer.T, dscores)
        grads['b2'] = np.sum(dscores, axis=0)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        grads['W1'] = np.dot(x_re.T, dhidden)
        grads['b1'] = np.sum(dhidden, axis=0)
        '''
        
        loss, dscores = softmax_loss(scores, y)
        loss_reg = 0.5*reg * np.sum(W2 * W2) + 0.5*reg * np.sum(W1 * W1)
        loss += loss_reg
        
        dh, dW2, db2 = affine_backward(dscores, cache2)
        dx, dW1, db1 = affine_relu_backward(dh, cache1)
        
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        # add regularization gradient contribution
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.hidden_dims = hidden_dims

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        L = len(hidden_dims)
        for i in range(L+1):
            # W1,b1
            if i==0:
                self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dims[i])
                self.params['b1'] = np.zeros(hidden_dims[0])
                if use_batchnorm:
                    self.params['gamma1'] = np.ones(hidden_dims[i])
                    self.params['beta1'] = np.zeros(hidden_dims[i])
            # W3,b3
            elif i==(L):
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(hidden_dims[i-1],num_classes)
                self.params['b'+str(i+1)] = np.zeros(num_classes)
            # w2,b2    
            else:
                self.params['W'+str(i+1)] = weight_scale*np.random.randn(hidden_dims[i-1],hidden_dims[i])
                self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
                if use_batchnorm:
                    self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                    self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        N = X.shape[0]
        D = np.prod([d for d in X.shape[1:]]) 
        num_train = X.shape[0]
        reg = self.reg
        
        # Unpack variables from the params dictionary
        L = len(self.hidden_dims)
        W=[0]
        b=[0]
        out=[0]
        cache=[0]
        dout=[0]
        '''
        out=np.zeros(L+1)
        cache=np.zeros(L+1)
        dout=np.zeros(L+1)
        '''
        for i in range(L+1):
            W.append(self.params['W'+str(i+1)])
            b.append(self.params['b'+str(i+1)])
        if self.use_batchnorm:
            gamma=[0]
            beta=[0]
            for i in range(L):
                gamma.append(self.params['gamma'+str(i+1)])
                beta.append(self.params['beta'+str(i+1)])
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        x_re = X.reshape(N,D)
        '''
        # ручная реализация
        hidden_layer.append(np.maximum(0, x_re.dot(W[1]) + b[1])) # ReLU activation 
        for i in range(L-1):
            hidden_layer.append(np.maximum(0, hidden_layer[i+1].dot(W[i+2]) + b[i+2])) # ReLU activation  
        scores = hidden_layer[L].dot(W[L+1]) + b[L+1]
        '''
        #out[1], cache[1] = affine_relu_forward(x_re, W[1], b[1])
        if self.use_batchnorm:
            out_el, cache_el = affine_batchnorm_relu_forward(x_re, W[1], b[1], gamma[1], beta[1], self.bn_params[0])
        else:
            out_el, cache_el = affine_relu_forward(x_re, W[1], b[1])
        out.append(out_el)
        cache.append(cache_el)
        
        for i in range(L-1):
            #out[i+2], cache[i+2] = affine_relu_forward(out[i+1], W[i+2], b[i+2])
            if self.use_batchnorm:
                out_el, cache_el = affine_batchnorm_relu_forward(out[i+1], W[i+2], b[i+2], gamma[i+2], beta[i+2], self.bn_params[i+1])
            else:    
                out_el, cache_el = affine_relu_forward(out[i+1], W[i+2], b[i+2])
            out.append(out_el)
            cache.append(cache_el)
            
        #scores, cache[L+1] = affine_forward(out[L], W[L+1], b[L+1])
        '''
        if self.use_batchnorm:
            scores, cache_el = affine_batchnorm_relu_forward(out[L], W[L+1], b[L+1], gamma[L+1], beta[L+1], self.bn_params[L])
        else:
            scores, cache_el = affine_forward(out[L], W[L+1], b[L+1])
        '''
        scores, cache_el = affine_forward(out[L], W[L+1], b[L+1])
        cache.append(cache_el)
        
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        '''
        # ручная реализация
        scores -= np.max(scores, axis=1, keepdims=True)  #to avoid numerical instability
        scores_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = np.exp(scores)/scores_exp
        loss = np.sum(-np.log(softmax[np.arange(num_train), y]))
        loss /= num_train
        loss_reg = 0
        for i in range(L+1):
            loss_reg += 0.5*reg * np.sum(W[L+1-i] * W[L+1-i])
        loss += loss_reg
        
        # compute the gradient on scores
        dscores = softmax
        dscores[range(num_train),y] -= 1
        dscores /= num_train
        
        # W3,b3
        grads['W'+str(L+1)] = np.dot(hidden_layer[L].T, dscores) + reg*W[L+1]
        grads['b'+str(L+1)] = np.sum(dscores, axis=0)
        dhidden = np.dot(dscores, W[L+1].T)
        dhidden[hidden_layer[L] <= 0] = 0
        # W2,b2
        for i in range(L-1):
            grads['W'+str(L-i)] = np.dot(hidden_layer[L-i-1].T, dhidden) + reg*W[L-i]
            grads['b'+str(L-i)] = np.sum(dhidden, axis=0)
            dhidden = np.dot(dhidden, W[L-i].T)  
            dhidden[hidden_layer[L-i-1] <= 0] = 0
            
        # W1,b1
        grads['W1'] = np.dot(x_re.T, dhidden) + reg*W[1]
        grads['b1'] = np.sum(dhidden, axis=0)
        '''
        # loss
        loss, dscores = softmax_loss(scores, y)
        loss_reg = 0
        for i in range(L+1):
            loss_reg += 0.5*reg * np.sum(W[L+1-i] * W[L+1-i])
        loss += loss_reg
        
        # W3,b3
        #dout[L], dW, db = affine_backward(dscores, cache[L+1])
        '''
        if self.use_batchnorm:
            dout_el, dW, db = affine_batchnorm_relu_backward(dscores, cache[L+1])
        else:    
            dout_el, dW, db = affine_backward(dscores, cache[L+1])
        '''
        dout_el, dW, db = affine_backward(dscores, cache[L+1])
        dout.append(dout_el)
        
        grads['W'+str(L+1)] = dW + reg*W[L+1]
        grads['b'+str(L+1)] = db
        
        # W2,b2; W1,b1
        for i in range(L):
            #dout[L-i-1], dW, db = affine_relu_backward(dout[L-i], cache[L-i])
            if self.use_batchnorm:
                dout_el, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout[i+1], cache[L-i])
                grads['gamma'+str(L-i)] = dgamma
                grads['beta'+str(L-i)] = dbeta
            else:    
                dout_el, dW, db = affine_relu_backward(dout[i+1], cache[L-i])
            dout.append(dout_el)
            grads['W'+str(L-i)] = dW + reg*W[L-i]
            grads['b'+str(L-i)] = db
          
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
