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
train_ds, train_info = tfds.load('horses_or_humans', split='train', with_info=True)
test_ds, test_info = tfds.load('horses_or_humans', split='test', with_info=True)

# Testing data
ds = train_ds.take(10)
for image, label in tfds.as_numpy(ds):
    if label == 0:
        print("Image is a horse")
    elif label == 1:
        print("Image is a human")
    else:
        print(ds)
        break

# Data Visualization
# fig = tfds.show_examples(train_ds, train_info)

# Metadata
m_train = train_info.splits["train"].num_examples
m_test = test_info.splits["test"].num_examples
num_px = train_info.features["image"].shape[1]

# Reshape the training ad test examples

# Turn tensor into numpy database
train_set_x_orig = []
test_set_x_orig = []


ds = train_ds.take(m_train)
for image, label in tfds.as_numpy(ds):
    train_set_x_orig.append(image)

ds = test_ds.take(m_test)
for image, label in tfds.as_numpy(ds):
    test_set_x_orig.append(image)

# Flatten Data