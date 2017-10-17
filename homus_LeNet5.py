'''
Trains a simple LeNet-5 (http://yann.lecun.com/exdb/lenet/) adapted to the HOMUS dataset using Keras Software (http://keras.io/)

LeNet-5 demo example http://eblearn.sourceforge.net/beginner_tutorial2_train.html

This example executed with 8x8 reescaled images and 50 epochs obtains an accuracy close to 36%.
'''

from __future__ import print_function
from PIL import Image, ImageOps
import numpy as np
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, adam, adadelta
from keras.models import load_model

np.random.seed(1337)  # for reproducibility

batch_size = 128
nb_classes = 32
nb_epoch = 50

# HOMUS contains images of 40 x 40 pixels
# input image dimensions for train 
img_rows, img_cols = 8, 8

# number of convolutional filters to use
nb_filters1 = 6
nb_filters2 = 16
nb_filters3 = 120

# convolution kernel size
nb_conv1 = 5
nb_conv2 = 6
nb_conv3 = 1

# size of pooling area for max pooling
nb_pool = 2
	
#
# Load data from data/HOMUS/train_0, data/HOMUS/train_1,...,data/HOMUS_31 folders from HOMUS images
#
def load_data():
	image_list = []
	class_list = []
	for current_class_number in range(0,nb_classes):	# Number of class
		for filename in glob.glob('./data/HOMUS/train_{}/*.jpg'.format(current_class_number)):
			im=Image.open(filename).resize((img_rows,img_cols)).convert('L')
			im=ImageOps.invert(im)	# Meaning of grey level is 255 (black) and 0 (white)
			#im.show()
			image_list.append(np.asarray(im).astype('float32')/255)
			class_list.append(current_class_number)

	n = len(image_list)	# Total examples			
	X = np.asarray(image_list).reshape(n,1,img_rows,img_cols)
	Y = np_utils.to_categorical(np.asarray(class_list), nb_classes)
	
	# Shuffle (X,Y)
	combined = zip(X, Y)
	np.random.shuffle(combined)
	X[:], Y[:] = zip(*combined)

	n_partition = int(n*0.9)	# Train 90% and Test 10%

	X_train = X[:n_partition]
	Y_train = Y[:n_partition]
	
	X_test  = X[n_partition:]
	Y_test  = Y[n_partition:]
	
	return X_train, Y_train, X_test, Y_test
	
# the data split between train and test sets
X_train, Y_train, X_test, Y_test = load_data()

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(img_rows,'x', img_cols, 'image size')
print(nb_epoch,'epochs')

#
# Neural Network Structure
#

model = Sequential()

model.add(Convolution2D(nb_filters1, nb_conv1, nb_conv1, border_mode='same', input_shape = (1, img_rows, img_cols)))
model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Activation("sigmoid"))

model.add(Convolution2D(nb_filters2, nb_conv2, nb_conv2, border_mode='same'))
model.add(AveragePooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Activation("sigmoid"))
model.add(Dropout(0.5))

model.add(Convolution2D(nb_filters3, nb_conv3, nb_conv3, border_mode='same'))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("sigmoid"))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

op=SGD(lr=0.8, momentum=0.8, decay=0.001, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=op,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_split=0.2)
score = model.evaluate(X_test, Y_test, verbose=0)

#
# Results
#

print('Test score:', score[0])
print('Test accuracy:', score[1])


# file name to save model
filename='homus_cnn.h5'

# save network model
model.save(filename)

# load neetwork model
#model = load_model(filename)

