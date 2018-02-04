import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
  # Initialize the loss and gradient to zero.
    loss = 0.0
    scores = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)  #to avoid numerical instability
        correct_class_score = scores[y[i]]
        scores_exp = 0.0
        softmax = 0.0
        for j in xrange(num_classes):
            scores_exp += np.exp(scores[j])
        softmax = np.exp(correct_class_score)/scores_exp    
        loss += -np.log(softmax)

        for k in xrange(num_classes):
            softmax_dW = np.exp(scores[k])/scores_exp
            dW[:,k] += (softmax_dW - (k == y[i]))*X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    # average
    loss /= num_train
    dW /= num_train
    # regularization
    loss += 0.5*reg * np.sum(W * W)
    dW += reg*W
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)  #to avoid numerical instability
    scores_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores)/scores_exp
    loss = np.sum(-np.log(softmax[np.arange(num_train), y]))

    indices = np.zeros_like(softmax)
    indices[np.arange(num_train),y] = 1
    dW = X.T.dot(softmax-indices)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    # average
    loss /= num_train
    dW /= num_train
    # regularization
    loss += 0.5*reg * np.sum(W * W)
    dW += reg*W
    return loss, dW

