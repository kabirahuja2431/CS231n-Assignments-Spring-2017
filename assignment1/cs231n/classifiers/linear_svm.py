import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in xrange(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      margin = scores - correct_class_score + 1
      margin[y[i]] = 0
      one = np.zeros(dW.shape)
      for j in xrange(len(margin)):
          if(margin[j] > 0):
              one[:,j] = np.ones(W.shape[0])
              dW[:,j] += X[i]
          else:
              one[:,j] = np.zeros(W.shape[0])
      dW[:,y[i]] += -1*np.sum(one,1)*X[i]
  
  dW/= num_train  
  #Adding regularization term in gradient
  dW += 2*reg*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  W_trans = W.transpose()
  correct_W = W_trans[y]
  scores = np.dot(X,W)
  func = scores - np.sum(X*correct_W,1).reshape((X.shape[0],1))+ 1                                       
  func[func < 0] = 0
    
  func[np.arange(func.shape[0]),y] = 0
       
  loss = np.sum(func)/X.shape[0]   + reg * np.sum(W * W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  func[func>0] = 1
  row_sum = np.sum(func,1)
  func[np.arange(func.shape[0]),y] = -1*row_sum.T  
  dW = np.dot(X.T,func)    
  dW/=X.shape[0]
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
