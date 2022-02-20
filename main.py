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

train_set_y = []           #Holds the label value of training set 0 is horse and 1 human
test_set_y = []            #Holds the label value of testing set 0 is horse and 1 human

ds = train_ds.take(m_train)
for image, label in tfds.as_numpy(ds):
    train_set_x_orig.append(image)
    train_set_y.append(label)

ds = test_ds.take(m_test)
for image, label in tfds.as_numpy(ds):
    test_set_x_orig.append(image)
    test_set_y.append(label)

train_set_x_orig = np.array(train_set_x_orig)
test_set_x_orig = np.array(test_set_x_orig)

train_set_y = np.array(train_set_y)
test_set_y = np.array(test_set_y)

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

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X)+b)                                                           #Compute Activation
    cost = (-1/m)*np.sum(np.dot(np.log(A), Y.T)+np.dot(np.log(1-A), (1-Y).T))               #Compute Cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {
        "dw": dw,
        "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations,learning_rate, print_cost = False):
    """
    Arguments:
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of shape (num_px * num_px * 3, numbers of examples)
    :param Y: true "label" vector (containing 0 if horse, 1 if human), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps5

    Returns:
    :return params: dictionary containing the weights w and bias b
    :return grads: dictionary containing the gradients of the weights and bias with respect to the cost function
    :return costs: list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    """
    Arguments --
    :param w: weights, a numpy array of size (num_px * num_px *3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px *3, number of examples)

    Returns --
    :return Y_prediction: a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a human/horse being present in the picture
    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0, i] to actual predictions p[0,1]
        if (A[0, i] > 0.5):
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

        assert(Y_prediction.shape == (1, m))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the functions implemented previously

    Arguments:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:

    Returns:
    :return d: dictionary containing information about the model
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = False)

    # Retrieve parameters w and b dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d



d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1000, learning_rate = 0.005, print_cost = True)
