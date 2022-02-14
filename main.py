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
m_train = train_info.splits["train"].num_examples
m_test = test_info.splits["test"].num_examples
num_px = train_info.features["image"].shape[1]

# Reshape the training ad test examples

# Turn tensor into numpy vectors of four arrays
    #lists
train_set_x_orig = []     #Holds the actual picture value of training set
test_set_x_orig = []      #Holds the actual picture value of testing set

train_set_y= []           #Holds the label value of training set 0 is horse and 1 human
test_set_y= []            #Holds the label value of testing set 0 is horse and 1 human

for image, label in tfds.as_numpy(train_ds):
    train_set_x_orig.append(image)
    train_set_y.append(label)

for image, label in tfds.as_numpy(test_ds):
    test_set_x_orig.append(image)
    test_set_y.append(label)

    #Vector
train_set_x_orig = np.array(train_set_x_orig)
test_set_x_orig = np.array(test_set_x_orig)

train_set_y= np.array(train_set_y)
test_set_y= np.array(test_set_y)

# Flatten Data
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

#Check data shape
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))