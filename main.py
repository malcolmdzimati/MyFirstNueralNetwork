# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
import tkinter

from PIL import Image
from scipy import ndimage

# Returns both train and test split separately
train_ds, train_info = tfds.load('horses_or_humans', split='train', with_info=True, as_supervised=True)
test_ds, test_info= tfds.load('horses_or_humans', split='test', with_info=True, as_supervised=True)

# Testing data
ds = test_ds.take(1)
for image, label in tfds.as_numpy(ds):
    if label == 0:
        print("Image is a horse")
    elif label == 1:
        print("Image is a human")
    else:
        print(ds)
        break

# Data Visualization
#fig = tfds.show_examples(train_ds, train_info)

# Metadata
m_train = 300                 #train_info.splits["train"].num_examples: 1027
m_test = 100                  #test_info.splits["test"].num_examples: 256
num_px = train_info.features["image"].shape[1]

# Reshape the training ad test examples

# Turn tensor into numpy vectors of four arrays
    #lists
train_set_x_orig = []     #Holds the actual picture value of training set
test_set_x_orig = []      #Holds the actual picture value of testing set

train_set_y= []           #Holds the label value of training set 0 is horse and 1 human
test_set_y= []            #Holds the label value of testing set 0 is horse and 1 human

ds =train_ds.take(m_train)
for image, label in tfds.as_numpy(ds):
    train_set_x_orig.append(image)
    train_set_y.append(label)

ds = test_ds.take(m_test)
for image, label in tfds.as_numpy(ds):
    test_set_x_orig.append(image)
    test_set_y.append(label)

train_set_x_orig = np.array(train_set_x_orig)
test_set_x_orig = np.array(test_set_x_orig)

train_set_y= np.array(train_set_y)
test_set_y= np.array(test_set_y)

# Flatten Data
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

"""Check data shape
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
"""

#Clearing Unused variables
train_set_x_orig = None
test_set_x_orig = None
train_ds = None
test_ds = None
train_info = None
test_info = None
ds = None

#Standardize dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

"""Sigmoid function <Activation Function>"""

def sigmoid(z):
    """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """
    s = 1/(1+np.exp(-z))

    return s

def initialize_with_zeros(dim):
    """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

def propagate(w, b, X, Y):
    """
        Implement the cost function and its gradient for the propagation

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        """
    m = X.shape[1]
    return


