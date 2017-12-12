import numpy as np
import glob
import tkinter
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, adam, adadelta
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# load neetwork model
filename = 'homus_cnn.h5'

model = load_model(filename)

print(model.evaluate())