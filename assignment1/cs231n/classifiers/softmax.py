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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  dim = X.shape[1]
  for i in xrange(num_train):
      scores = X[i].dot(W)
      C = -1*np.max(scores)
      exp_scores = np.exp(scores+C)
      current_exp_score = exp_scores[y[i]]
      exp_loss = current_exp_score/np.sum(exp_scores)
      loss+=-1*np.log(exp_loss)
      prob = exp_scores/np.sum(exp_scores)
      prob = prob.reshape(1,prob.shape[0])
      x = X[i].reshape(dim,1)
      dW+=np.dot(x,prob)
      dW[:,y[i]]-=X[i]
  loss/=num_train
  loss+=reg*np.sum(W*W)
  dW/=num_train
  dW+=2*reg*W
  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  scores = np.dot(X,W)
  C= np.max(scores,1).reshape(num_train,1)
  exp_scores = np.exp(scores-C)
  W_trans = W.transpose()
  correct_W = W_trans[y]
  correct_scores = np.sum(X*correct_W,1)
  correct_exp_scores = np.exp(correct_scores-C.reshape(num_train,))
  loss += -1*np.sum(np.log(correct_exp_scores/np.sum(exp_scores,1)))
  loss/=num_train
  loss+=reg*np.sum(W*W)
  
  prob = exp_scores.T/np.sum(exp_scores,1)
  prob = prob.T
  prob[np.arange(prob.shape[0]),y] -= 1
  dW = (X.T).dot(prob)
  dW/=num_train
  dW+=2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

