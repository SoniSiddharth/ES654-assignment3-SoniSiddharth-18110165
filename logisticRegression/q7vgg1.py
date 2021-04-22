import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import os
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folder = 'C:/Users/user/Desktop/semester6/mlES654/Assignment3/assignment-3-SoniSiddharth/logisticRegression/images/'
photos, labels = list(), list()


# for file in listdir(folder):
# 	print(1)
# 	# determine class
# 	output = 0.0
# 	# print(file)
# 	if file.startswith('sloth'):
# 		output = 1.0
# 	# load image
# 	photo = load_img(folder + file, target_size=(200, 200))
# 	# convert to numpy array
# 	photo = img_to_array(photo)
# 	# store
# 	photos.append(photo)
# 	labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('sloth_and_snake_photos.npy', photos)
# save('sloth_and_snake_labels.npy', labels)



from numpy import load
photos = load('sloth_and_snake_photos.npy')
labels = load('sloth_and_snake_labels.npy')
print(photos.shape, labels.shape)

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories
# dataset_home = 'images/'
# subdirs = ['train/', 'test/']
# for subdir in subdirs:
# 	# create label subdirectories
# 	labeldirs = ['sloths/', 'snakes/']
# 	for labldir in labeldirs:
# 		newdir = dataset_home + subdir + labldir
# 		os.makedirs(newdir, exist_ok=True)

# import random
# sloth_images = []
# snake_images = []

# for file in listdir(folder):
# 	if file.startswith('sloth'):
# 		sloth_images.append(file)
# 	elif file.startswith('snake'):
# 		snake_images.append(file)

# random.seed(42)
# random.shuffle(sloth_images) # shuffles the ordering of filenames (deterministic given the chosen seed)
# random.shuffle(snake_images) # shuffles the ordering of filenames (deterministic given the chosen seed)

# split_1 = int(0.25 * len(sloth_images))

# test_sloths = sloth_images[:split_1]
# train_sloths = sloth_images[split_1:]

# test_snakes = snake_images[:split_1]
# train_snakes = snake_images[split_1:]

# print(len(train_sloths))
# print(len(test_sloths))
# print(train_sloths)

# src_directory = 'images/'
# dst_dir = 'train/'
# for j in range(len(train_sloths)):
# 	src = src_directory + train_sloths[j]
# 	dst = dataset_home + dst_dir + 'sloths/' + train_sloths[j]
# 	copyfile(src, dst)

# 	src = src_directory + train_snakes[j]
# 	dst = dataset_home + dst_dir + 'snakes/' + train_snakes[j]
# 	copyfile(src, dst)


# dst_dir = 'test/'
# for j in range(len(test_sloths)):
# 	src = src_directory + test_sloths[j]
# 	dst = dataset_home + dst_dir + 'sloths/' + test_sloths[j]
# 	copyfile(src, dst)

# 	src = src_directory + test_snakes[j]
# 	dst = dataset_home + dst_dir + 'snakes/' + test_snakes[j]
# 	copyfile(src, dst)


def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()


# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
 
# run the test harness for evaluating a model
def run_test_harness():
	# define model
	model = define_model()
	# create data generator
	datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
	train_it = datagen.flow_from_directory('images/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	test_it = datagen.flow_from_directory('images/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history)
 
# entry point, run the test harness
run_test_harness()