from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from collections import defaultdict
import re
#from bs4 import BeautifulSoup
import sys
import os
import time
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda, Reshape, Dot, Permute, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer
#from make_data import generate_data
import json
import random
from keras import optimizers
import argparse

from tensorflow.python import debug as tf_debug

from make_data import generate_data
from keras.datasets import mnist
from scipy.misc import imsave
from tqdm import tqdm

# No of Groups
num_groups = 5

BATCH_SIZE = 256
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
# The number of key features for each data set.
ks = {'orange_skin': 1, 'XOR': 1, 'nonlinear_additive': 1, 'switch': 1}


def create_data(downsampling=False):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# downsampling
	if downsampling:
		x_train = x_train[:, ::2, ::2]
		x_test = x_test[:, ::2, ::2]
	x_train = np.expand_dims(x_train, -1)
	x_test = np.expand_dims(x_test, -1)
	y_train = np.eye(10)[y_train]
	y_test = np.eye(10)[y_test]
	input_shape = x_train.shape[1:]
	return x_train,y_train,x_test,y_test,input_shape


def create_rank(scores, k):
	"""
	Compute rank of each feature based on weight.

	"""
	scores = abs(scores)
	n, d = scores.shape
	ranks = []
	for i, score in enumerate(scores):
		# Random permutation to avoid bias due to equal weights.
		idx = np.random.permutation(d)
		permutated_weights = score[idx]
		permutated_rank=(-permutated_weights).argsort().argsort()+1
		rank = permutated_rank[np.argsort(idx)]

		ranks.append(rank)

	return np.array(ranks)

def compute_median_rank(scores, k, datatype_val = None):
	ranks = create_rank(scores, k)
	if datatype_val is None:
		median_ranks = np.median(ranks[:,:k], axis = 1)
	else:
		datatype_val = datatype_val[:len(scores)]
		median_ranks1 = np.median(ranks[datatype_val == 'orange_skin',:][:,np.array([0,1,2,3,9])],
			axis = 1)
		median_ranks2 = np.median(ranks[datatype_val == 'nonlinear_additive',:][:,np.array([4,5,6,7,9])],
			axis = 1)
		median_ranks = np.concatenate((median_ranks1,median_ranks2), 0)
	return median_ranks

def compute_groups(scores):
	# TODO: implementation needed
	return None

class Sample_Concrete(Layer):
	"""
	Layer for sample Concrete / Gumbel-Softmax variables.
	"""
	def __init__(self, tau0, k, num_feature, num_groups, **kwargs):
		self.tau0 = tau0
		self.k = k
		self.num_groups = num_groups
		self.num_feature = num_feature
		super(Sample_Concrete, self).__init__(**kwargs)

	def call(self, logits):
		# logits: [BATCH_SIZE, num_feature * num_groups]
		logits_ = K.expand_dims(logits, -2)# [BATCH_SIZE, 1, num_feature * num_groups]
		batch_size = tf.shape(logits_)[0]
		num_feature = self.num_feature
		num_groups = self.num_groups
		samples_list = []
		discrete_logits_list = []
		for i in range(num_feature):
			#sub_logits = logits_[:,:,i*num_feature:(i+1)*num_feature]

			sub_logits = logits_[:,:,i*num_groups:(i+1)*num_groups]			


			uniform = tf.random_uniform(shape =(batch_size, self.k, num_groups),
				minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
				maxval = 1.0)

			gumbel = - K.log(-K.log(uniform))
			noisy_logits = (gumbel + sub_logits)/self.tau0
			samples = K.softmax(noisy_logits)
			samples = K.max(samples, axis = 1)

			# Explanation Stage output.
			threshold = tf.expand_dims(tf.nn.top_k(logits[:,i*num_groups:(i+1)*num_groups], self.k, sorted = True)[0][:,-1], -1)
			discrete_logits = tf.cast(tf.greater_equal(logits[:,i*num_groups:(i+1)*num_groups],threshold),tf.float32)

			samples_list.append(samples)
			discrete_logits_list.append(discrete_logits)

		final_samples = tf.concat(samples_list, 1)
		final_discrete_logits = tf.concat(discrete_logits_list, 1)



		return K.in_train_phase(final_samples, final_discrete_logits)

	def compute_output_shape(self, input_shape):
		return input_shape



def L2X(datatype="mnist", train = True):
	# the whole thing is equation (5)
	x_train,y_train,x_val,y_val, input_shape = create_data()

	st1 = time.time()
	st2 = st1
	print (input_shape)
	activation = 'relu'
	# P(S|X) we train the model on this, for capturing the important features.
	model_input = Input(shape=input_shape, dtype='float32')

	net = Conv2D(8, (3, 3), activation='relu', padding='same')(model_input)
	net = MaxPooling2D((2, 2), padding='same')(net)
	net = Conv2D(8, (3, 3), activation='relu', padding='same')(net)
	net = MaxPooling2D((2, 2), padding='same')(net)
	net = Conv2D(1, (3, 3), activation='relu', padding='same')(net)
	net = Flatten()(net) # now 49

	'''net = Dense(256, activation=activation, name = 's/dense1',
		kernel_regularizer=regularizers.l2(1e-3))(model_input)
	net = Dense(256, activation=activation, name = 's/dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)'''

	# A tensor of shape, [batch_size, max_sents, 100]

	mid_dim = 49 * num_groups


	logits = Dense(mid_dim)(net)
	# [BATCH_SIZE, max_sents, 1]

	k = 1; tau = 0.1

	samples = Sample_Concrete(tau, k, 49, num_groups, name = 'sample')(logits)

	# samples = Reshape((num_groups, input_shape))(samples)
	samples = Reshape((49, num_groups))(samples)
	samples = Permute((2, 1))(samples) 
	'''def get_shape(input_shapes):
		return tuple((input_shapes[0], num_groups, input_shapes[1]))
	tile_layer = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis = -2), [1, num_groups, 1]), output_shape=get_shape)
	model_input_tiled = tile_layer(model_input)
	#print(model_input_tiled.shape)
	new_input = Multiply()([model_input_tiled, samples])
	get_first = Lambda(lambda x: x[:, 0, :], output_shape=lambda x: tuple((x[0], x[2])))
	new_model_input = get_first(new_input)'''
	#new_model_input = Flatten()(new_input)
	#samples to be KD *1 and then make a matrix K*D and the K*D * D * 1 = K * 1 the new_model_input
	#   1) one nueral net that gives
	#   2) seperate neural net with one node as input.

	# q(X_S) variational family
	# new_model_input = Multiply()([model_input, samples])
	# new_model_input =  Dot(samples, model_input)

	def matmul_output_shape(input_shapes):
	    shape1 = list(input_shapes[0])
	    shape2 = list(input_shapes[1])
	    return tuple((shape1[0], shape1[1]))
    
	matmul_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]), output_shape=matmul_output_shape)
	new_model_input = matmul_layer([samples, net])

	print('heihei', net.shape)
	net = Dense(32, activation=activation, name = 'dense1',
		kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
	net = BatchNormalization()(net) # Add batchnorm for stability.
	net = Dense(16, activation=activation, name = 'dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	net = BatchNormalization()(net)

	preds = Dense(10, activation='softmax', name = 'dense4',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	model = Model(model_input, preds)
	model.summary()

	if train:
		adam = optimizers.Adam(lr = 1e-3)
		model.compile(loss='categorical_crossentropy',
					  optimizer=adam,
					  metrics=['acc'])
		filepath="models/{}/L2X.hdf5".format(datatype)
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
			verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=5, batch_size=BATCH_SIZE)
		st2 = time.time()
	else:
		model.load_weights('models/{}/L2X.hdf5'.format(datatype),
			by_name=True)


	pred_model = Model(model_input, samples)
	#pred_model.compile(loss=None,
	#				  optimizer='rmsprop',
	#			  metrics=[None])

	# For now samples is a matrix instead of a vector
	
	scores = pred_model.predict(x_val, verbose = 1, batch_size = BATCH_SIZE)


	# We need to write a new compute_median_rank to do analysis
	# median_ranks = compute_median_rank(scores, k = ks[datatype],
	#		datatype_val=datatype_val)
	median_ranks = compute_groups(scores)

	return median_ranks, time.time() - st2, st2 - st1, scores,x_val,y_val

def visualize_group(score, output_path):
	colors = [(255,0,0), (0,255,0), (0,0,255), 	(255,255,0), (0,255,255), (255,0,255), (255,255,255), 	(0,0,0), (128,0,0), (0,128,128)]
	score_img = np.zeros((49, 3), dtype=np.uint8)

	num_grp, num_ft = score.shape
	assert num_ft == 49
	for i in range(num_grp):
		place = np.where(score[i] > 0)
		score_img[place] = colors[i]
	score_img = np.resize(score_img, (7, 7, 3))
	#print(score_img.shape)
	imsave(output_path, score_img)


if __name__ == '__main__':
	#sess = K.get_session()
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)	
	#K.set_session(sess)
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')

	args = parser.parse_args()

	median_ranks, exp_time, train_time,scores,x_val,y_val = L2X(train = args.train)
	output = 'datatype:{}, mean:{}, sd:{}, train time:{}s, explain time:{}s \n'.format(
		"mnist",
		0, #np.mean(median_ranks),
		0, #np.std(median_ranks),
		train_time, exp_time)
	print (scores)
	pickle.dump(scores, open("./score.pkl", "wb"))
	pickle.dump(x_val, open("./x_val.pkl", "wb"))
	pickle.dump(y_val, open("./y_val.pkl", "wb"))
	num_val = scores.shape[0]
	for num in tqdm(range(100)):
		visualize_group(scores[num], 'img_groups/group_%03d.png' % num)
		imsave('img_numbers/num_%03d.png' % num, np.squeeze(x_val[num]).astype(np.uint8))
	print(output)
