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
from keras.layers import Dense, Input, Flatten, Add, Multiply, Lambda, Reshape, Dot
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



# No of Groups
num_groups = 4

BATCH_SIZE = 256
np.random.seed(0)
tf.set_random_seed(0)
random.seed(0)
# The number of key features for each data set.
ks = {'RNA' : 1}

def create_data(datatype, n = 1000):
	dir_res = "./Results200/"


	y_train = pickle.load(open(dir_res+"training_data.pkl", "rb"))
	x_train = pickle.load(open(dir_res+"y_ns_training.pkl", "rb"))
	y_val = pickle.load(open(dir_res+"test_data.pkl", "rb"))
	x_val = pickle.load(open(dir_res+"y_ns_test.pkl", "rb"))



	mean = x_train.mean(axis=0)

	std = x_train.std(axis=0)
	x_train -= mean
	x_train /= std
	x_val -= mean
	x_val /= std
	datatypes_val = None

	input_shape = x_train.shape[1]

	return x_train,y_train,x_val,y_val,datatypes_val,input_shape


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
		for i in range(num_groups):
			sub_logits = logits_[:,:,i*num_feature:(i+1)*num_feature]

			uniform = tf.random_uniform(shape =(batch_size, self.k, num_feature),
				minval = np.finfo(tf.float32.as_numpy_dtype).tiny,
				maxval = 1.0)

			gumbel = - K.log(-K.log(uniform))
			noisy_logits = (gumbel + sub_logits)/self.tau0
			samples = K.softmax(noisy_logits)
			samples = K.max(samples, axis = 1)

			# Explanation Stage output.
			threshold = tf.expand_dims(tf.nn.top_k(logits[:,i*num_feature:(i+1)*num_feature], self.k, sorted = True)[0][:,-1], -1)
			discrete_logits = tf.cast(tf.greater_equal(logits[:,i*num_feature:(i+1)*num_feature],threshold),tf.float32)

			samples_list.append(samples)
			discrete_logits_list.append(discrete_logits)

		final_samples = tf.concat(samples_list, 1)
		final_discrete_logits = tf.concat(discrete_logits_list, 1)
		return K.in_train_phase(final_samples, final_discrete_logits)

	def compute_output_shape(self, input_shape):
		return input_shape



def L2X(datatype, train = True):
	# the whole thing is equation (5)
	x_train,y_train,x_val,y_val,datatype_val, input_shape = create_data(datatype,
		n = int(1e6))

	st1 = time.time()
	st2 = st1
	print (input_shape)
	activation = 'relu'
	# P(S|X) we train the model on this, for capturing the important features.
	model_input = Input(shape=(input_shape,), dtype='float32')

	net = Dense(100, activation=activation, name = 's/dense1',
		kernel_regularizer=regularizers.l2(1e-3))(model_input)
	net = Dense(100, activation=activation, name = 's/dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)

	# A tensor of shape, [batch_size, max_sents, 100]

	mid_dim = input_shape * num_groups


	logits = Dense(mid_dim)(net)
	# [BATCH_SIZE, max_sents, 1]

	k = ks[datatype]; tau = 0.1

	samples = Sample_Concrete(tau, k, input_shape, num_groups, name = 'sample')(logits)

	samples = Reshape((num_groups, input_shape))(samples)


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
	new_model_input = matmul_layer([samples, model_input])
	print('heihei', new_model_input.shape)
	net = Dense(32, activation=activation, name = 'dense1',
		kernel_regularizer=regularizers.l2(1e-3))(new_model_input)
	net = BatchNormalization()(net) # Add batchnorm for stability.
	net = Dense(16, activation=activation, name = 'dense2',
		kernel_regularizer=regularizers.l2(1e-3))(net)
	net = BatchNormalization()(net)

	preds = Dense(2, activation='softmax', name = 'dense4',
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
		model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=1000, batch_size=BATCH_SIZE)
		st2 = time.time()
	else:
		model.load_weights('models/{}/L2X.hdf5'.format(datatype),
			by_name=True)


	pred_model = Model(model_input, samples)
	pred_model.compile(loss=None,
				  optimizer='rmsprop',
				  metrics=[None])

	# For now samples is a matrix instead of a vector
	
	scores = pred_model.predict(x_val, verbose = 1, batch_size = BATCH_SIZE)

	# We need to write a new compute_median_rank to do analysis
	# median_ranks = compute_median_rank(scores, k = ks[datatype],
	#		datatype_val=datatype_val)
	median_ranks = compute_groups(scores)

	return median_ranks, time.time() - st2, st2 - st1, scores,x_val,y_val


if __name__ == '__main__':
	#sess = K.get_session()
	#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
	#K.set_session(sess)
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--datatype', type = str,
		choices = ['RNA'], default = 'RNA')
	parser.add_argument('--train', action='store_true')

	args = parser.parse_args()

	median_ranks, exp_time, train_time,scores,x_val,y_val = L2X(datatype = args.datatype,
		train = args.train)
	output = 'datatype:{}, mean:{}, sd:{}, train time:{}s, explain time:{}s \n'.format(
		args.datatype,
		0, #np.mean(median_ranks),
		0, #np.std(median_ranks),
		train_time, exp_time)
	print (scores)
	#pickle.dump(scores, open("./score.pkl", "wb"))
	#pickle.dump(x_val, open("./x_val.pkl", "wb"))
	#pickle.dump(y_val, open("./y_val.pkl", "wb"))

	print(output)
