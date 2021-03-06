"""
The code for constructing the original word-CNN is based on
https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
"""
from __future__ import absolute_import, division, print_function   
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Embedding, Dense, Dropout, Activation, Reshape, Permute, Flatten
from keras.datasets import imdb
from keras.engine.topology import Layer 
from keras import backend as K  
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential 

import numpy as np
import tensorflow as tf 
import time 
import numpy as np 
import sys
import os
#import urllib2 
#import tarfile
#import zipfile 
#try:
#    import cPickle as pickle
#except:
import pickle
import os 
from utils import create_dataset_from_score, calculate_acc, get_selected_words_group, create_dataset_from_group_score



# Set parameters:
tf.set_random_seed(10086)
np.random.seed(10086)
max_features = 5000
maxlen = 100
num_groups = 3
batch_size = 40
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 3
k =1 # Number of selected words by L2X.
PART_SIZE = 125

###########################################
###############Load data###################
###########################################

def load_data():
    """
    Load data if data have been created.
    Create data otherwise.

    """

    if 'data' not in os.listdir('.'):
        os.mkdir('data') 
        
    if 'id_to_word.pkl' not in os.listdir('data'):
        print('Loading data...')
        (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, skip_top=100, index_from=3)
        word_to_id = imdb.get_word_index()
        word_to_id ={k:(v+3) for k,v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2
        id_to_word = {value:key for key,value in word_to_id.items()}

        print(len(x_train), 'train sequences')
        print(len(x_val), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
        y_train = np.eye(2)[y_train]
        y_val = np.eye(2)[y_val] 

        np.save('./data/x_train.npy', x_train)
        np.save('./data/y_train.npy', y_train)
        np.save('./data/x_val.npy', x_val)
        np.save('./data/y_val.npy', y_val)
        with open('data/id_to_word.pkl','wb') as f:
            pickle.dump(id_to_word, f)  

    else:
        x_train, y_train, x_val, y_val = np.load('data/x_train.npy'),np.load('data/y_train.npy'),np.load('data/x_val.npy'),np.load('data/y_val.npy')
        with open('data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)

    return x_train, y_train, x_val, y_val, id_to_word

###########################################
###############Original Model##############
###########################################

def create_original_model():
    """
    Build the original model to be explained. 

    """
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model




def generate_original_preds(train = True): 
    """
    Generate the predictions of the original model on training
    and validation datasets. 

    The original model is also trained if train = True. 

    """
    x_train, y_train, x_val, y_val, id_to_word = load_data() 
    model = create_original_model()

    if train:
        filepath="models/original.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.fit(x_train, y_train, validation_data=(x_val, y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)

    model.load_weights('./models/original.hdf5', 
        by_name=True) 

    pred_train = model.predict(x_train,verbose = 1, batch_size = 1000)
    pred_val = model.predict(x_val,verbose = 1, batch_size = 1000)
    if not train:
        print('The val accuracy is {}'.format(calculate_acc(pred_val,y_val)))
        print('The train accuracy is {}'.format(calculate_acc(pred_train,y_train)))


    np.save('data/pred_train.npy', pred_train)
    np.save('data/pred_val.npy', pred_val) 

###########################################
####################L2X####################
###########################################
# Define various Keras layers.
Mean = Lambda(lambda x: K.sum(x, axis = 1) / float(k), 
    output_shape=lambda x: [x[0],x[2]]) 

def matmul_output_shape(input_shapes):
        shape1 = list(input_shapes[0])
        shape2 = list(input_shapes[1])
        return tuple((shape1[0], shape1[1], shape2[2]))
    
matmul_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]), output_shape=matmul_output_shape)

class Concatenate(Layer):
    """
    Layer for concatenation. 
    
    """
    def __init__(self, **kwargs): 
        super(Concatenate, self).__init__(**kwargs)

    def call(self, inputs):
        input1, input2 = inputs  
        input1 = tf.expand_dims(input1, axis = -2) # [batchsize, 1, input1_dim] 
        dim1 = int(input2.get_shape()[1])
        input1 = tf.tile(input1, [1, dim1, 1])
        return tf.concat([input1, input2], axis = -1)

    def compute_output_shape(self, input_shapes):
        input_shape1, input_shape2 = input_shapes
        input_shape = list(input_shape2)
        input_shape[-1] = int(input_shape[-1]) + int(input_shape1[-1])
        input_shape[-2] = int(input_shape[-2])
        return tuple(input_shape)

'''class Sample_Concrete(Layer):
    """
    Layer for sample Concrete / Gumbel-Softmax variables. 

    """
    def __init__(self, tau0, k, **kwargs): 
        self.tau0 = tau0
        self.k = k
        super(Sample_Concrete, self).__init__(**kwargs)

    def call(self, logits):   
        # logits: [batch_size, d, 1]
        logits_ = K.permute_dimensions(logits, (0,2,1))# [batch_size, 1, d]

        d = int(logits_.get_shape()[2])
        unif_shape = [batch_size,self.k,d]

        uniform = K.random_uniform_variable(shape=unif_shape,
            low = np.finfo(tf.float32.as_numpy_dtype).tiny,
            high = 1.0)
        gumbel = - K.log(-K.log(uniform))
        noisy_logits = (gumbel + logits_)/self.tau0
        samples = K.softmax(noisy_logits)
        samples = K.max(samples, axis = 1) 
        logits = tf.reshape(logits,[-1, d]) 
        threshold = tf.expand_dims(tf.nn.top_k(logits, self.k, sorted = True)[0][:,-1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits,threshold),tf.float32)
        
        output = K.in_train_phase(samples, discrete_logits) 
        return tf.expand_dims(output,-1)

    def compute_output_shape(self, input_shape):
        return input_shape'''
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

def construct_gumbel_selector(X_ph, num_words, embedding_dims, maxlen):
    """
    Build the L2X model for selecting words. 

    """
    emb_layer = Embedding(num_words, embedding_dims, input_length = maxlen, name = 'emb_gumbel')
    emb = emb_layer(X_ph) #(400, 50) 
    net = Dropout(0.2, name = 'dropout_gumbel')(emb)
    net = emb
    first_layer = Conv1D(100, kernel_size, padding='same', activation='relu', strides=1, name = 'conv1_gumbel')(net) # bs, 400, 100

    # global info
    net_new = GlobalMaxPooling1D(name = 'new_global_max_pooling1d_1')(first_layer) # bs, 100
    global_info = Dense(100, name = 'new_dense_1', activation='relu')(net_new) # bs, 100

    # local info
    net = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv2_gumbel')(first_layer) # bs, 400, 100
    local_info = Conv1D(100, 3, padding='same', activation='relu', strides=1, name = 'conv3_gumbel')(net)  # bs, 400, 100
    combined = Concatenate()([global_info,local_info]) 
    net = Dropout(0.2, name = 'new_dropout_2')(combined)
    net = Conv1D(100, 1, padding='same', activation='relu', strides=1, name = 'conv_last_gumbel')(net)   

    logits_T = Conv1D(1, 1, padding='same', activation=None, strides=1, name = 'conv4_gumbel')(net)  # bs, 400, 1
    # wanna make it bs, maxlen*num_groups
    squeeze_layer = Lambda(lambda x:tf.squeeze(x), output_shape=lambda x:x[:-1])

    logits_T_grp = Dense(maxlen*num_groups)(squeeze_layer(logits_T))
    #print(logits_T_grp.shape)
    return logits_T_grp # bs, 400* num_groups


def L2X(train = True): 
    """
    Generate scores on features on validation by L2X.

    Train the L2X model with variational approaches 
    if train = True. 

    """
    print('Loading dataset...') 
    x_train, y_train, x_val, y_val, id_to_word = load_data()
    #pred_train = np.load('data/pred_train.npy')
    #pred_val = np.load('data/pred_val.npy') 
    print('Creating model...')

    # P(S|X)
    with tf.variable_scope('selection_model'):
        X_ph = Input(shape=(maxlen,), dtype='int32')

        logits_T_grp = construct_gumbel_selector(X_ph, max_features, embedding_dims, maxlen) # bs, max_len *  num_groups
        tau = 0.5 
        T = Sample_Concrete(tau, k, num_feature=maxlen, num_groups=num_groups)(logits_T_grp)

        T = Reshape((maxlen, num_groups))(T)
        T = Permute((2, 1))(T) # bs, num_groups, max_len

    # q(X_S)
    with tf.variable_scope('prediction_model'):
        emb2 = Embedding(max_features, embedding_dims, 
            input_length=maxlen)(X_ph)
        # emb2 bs, max_len, 50
        # apply the matrix trick as before
        # here the output size of matmul layer is different from before
        net = matmul_layer([T, emb2]) # bs, num_groups, 50
        print(net.shape)


        # net = Mean(net) # bs, 50
        net = Flatten()(net)
        net = Dense(hidden_dims)(net)
        net = Activation('relu')(net) 
        preds = Dense(2, activation='softmax', 
            name = 'new_dense')(net)


    model = Model(inputs=X_ph, 
        outputs=preds)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',#optimizer,
                  metrics=['acc']) 
    #train_acc = np.mean(np.argmax(pred_train, axis = 1)==np.argmax(y_train, axis = 1))
    #val_acc = np.mean(np.argmax(pred_val, axis = 1)==np.argmax(y_val, axis = 1))
    #print('The train and validation accuracy of the original model is {} and {}'.format(train_acc, val_acc))

    if train:
        filepath="models/l2x.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
            verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint] 
        st = time.time()
        model.fit(x_train, y_train, 
            validation_data=(x_val, y_val), 
            callbacks = callbacks_list,
            epochs=epochs, batch_size=batch_size)
        duration = time.time() - st
        print('Training time is {}'.format(duration))       

    model.load_weights('models/l2x.hdf5', by_name=True) 

    pred_model = Model(X_ph, T) 
    pred_model.summary()
    pred_model.compile(loss='categorical_crossentropy', 
        optimizer='adam', metrics=['acc']) 

    st = time.time()
    #scores = pred_model.predict(x_val, 
    #    verbose = 1, batch_size = batch_size)[:,:,0] 
    #scores = np.reshape(scores, [scores.shape[0], maxlen])
    scores = pred_model.predict(x_val, verbose = 1, batch_size = batch_size)
    return scores, x_val 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = str, 
        choices = ['original','L2X'], default = 'original') 
    parser.add_argument('--train', action='store_true')  
    #parser.set_defaults(train=False)
    args = parser.parse_args()
    dict_a = vars(args)
    if not os.path.exists('./models'):
        os.mkdir('./models')

    if args.task == 'original':
        generate_original_preds(args.train) 

    elif args.task == 'L2X':
        scores, x = L2X(args.train)
        print(scores.shape) # batchs, num_groups, max_len
        print(scores[0])
        print('Creating dataset with selected sentences...')
        explain_list = create_dataset_from_group_score(x, scores, filtered=True)
        print(explain_list[0])
        print(explain_list[1])
        with open('./data/explain_list.pkl', 'wb') as f:
            pickle.dump(explain_list, f)
        # TODO: save scores and do analysis
        # create_dataset_from_score(x, scores, k)




