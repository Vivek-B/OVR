import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf 
import itertools
import os.path

import keras
from keras.layers import Dense,Dropout,Input,BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error
from keras import regularizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## Arguments for running the python script
parser = argparse.ArgumentParser(description='OVR-Encoder Training')
parser.add_argument('--n_comp', type=int,  help='No. of PCA components', required=False, default=781)
parser.add_argument('--encoded_size', type=int,  help='Encoded dimension', required=False, default=8192)
parser.add_argument('--lamda', type=float,  help='OVR regularization strength', required=False, default=0.00005)
parser.add_argument('--act_fn', type=str,  help='Activation function for encoder', required=False, default='relu')
parser.add_argument('--encoder_epochs', type=int,  help='Number of epochs for Encoder', required=False, default=35)
parser.add_argument('--batch_size', type=int,  help='Batch size', required=False, default=128)
parser.add_argument('--model_path', type=str,  help='Output path for saved model', required=False, default='model')


def load_cifar10_pca(n_comp=781, scale=True):
	'''	This function loads the cifar-10 data from keras
		Flattens the train and test images
		Does standardization
		Computes PCA and reduces dimensions

		Arguments:
			n_comp - No of components for PCA reduced data
			scale - Weather to standardize the data or not

		Returns:
			Train & Test data of cifar-10 data
	'''

	## Load Cifar-10 data from keras datasets
	print('\n')
	print('Loading cifar10 from keras ...')
	(cifar_train_data, train_labels), (cifar_test_data, test_labels) = cifar10.load_data();

	## Flatten the images 
	cifar_train_data = cifar_train_data.reshape(cifar_train_data.shape[0],-1)
	cifar_test_data = cifar_test_data.reshape(cifar_test_data.shape[0],-1)
	print('\n')
	print('After Flattening:')
	print(cifar_train_data.shape)
	print(cifar_test_data.shape)

	if scale==True:
		## Standardize the pixels
		scaler = StandardScaler(with_mean=True, with_std=True)
		scaler.fit(cifar_train_data)
		cifar_train_data = scaler.transform(cifar_train_data)
		cifar_test_data = scaler.transform(cifar_test_data)

	## Compute PCA  
	print('\n')
	print('Computing PCA on cifar10 ...')
	pca = PCA(n_components=cifar_train_data.shape[1], random_state=496, svd_solver='full', whiten=False)
	pca.fit(cifar_train_data)
	train_pca = pca.transform(cifar_train_data)
	test_pca = pca.transform(cifar_test_data)

	## Reduce the dimensions
	X = train_pca[:,:n_comp]
	test_X = test_pca[:,:n_comp]

	## Convert the target to one-hot vectors
	Y = np.asarray(train_labels,dtype=np.int32)
	Y = keras.utils.to_categorical(Y)
	test_Y = np.asarray(test_labels,dtype=np.int32)
	test_Y = keras.utils.to_categorical(test_Y)

	return(X,Y,test_X,test_Y)


def wrapper_loss(hiddens,lamda=0.00001,decoder=False):
	print('\n')
	print('Defining OVR Loss ...')
	''' Custom OVR Loss Function in Keras
		'hiddens_n' is normalized activations(hiddens)
		'ovr' corresponds to OVR loss. Sum of dot products between all activations in a mini-batch
		'loss2' is introduced to encourage non zero activations

		Arguments:
			hiddens - Activations from a fully-connected layer
			lamda - regularization strength for OVR cost
			decoder - Weather the decoder is trained or not
			For encoder-only model, set 'decoder' to False. 
			decoder==False, 'mse' is set to 0, which will not train the decoder part. Virtually making it an Encoder-Only model

		Returns:
			Custom Loss
	'''

	mse = 0 if decoder==False else 1

	hiddens_n = tf.divide(hiddens,tf.expand_dims(tf.norm(hiddens,2,1),axis=1))
	def custom_cost_function(y_true,y_pred):
		ovr = lamda*tf.reduce_sum(tf.matmul(hiddens_n,hiddens_n,transpose_b=True))
		loss2 = tf.abs(tf.reduce_mean(hiddens)-0.5)
		return ovr + loss2 + mse*mean_squared_error(y_true, y_pred)
	return custom_cost_function


## Define the Encoder in Keras
def autoencoder(inp_size,nhids,act_fn,lamda,weights_path=None):
	''' The is function build the AutoEncoder model.
		There's only one hidden layer
		For Encoder-Only mmodel, pass 'decoder'==False
		For AutoEncoder with ovr-regularization, pass 'decoder'==True

		Arguments:
			inp_size - No. input features
			nhids - Hidden layer size 
			act_fn - hidden layer activation
			lamda - regularization strength for OVR cost
			weights_path - if this argument is passed, pretrained weights are loaded. Otherwise, parameters are initialized randomly

		Returns:
			AutoEncoder, Encoder models 
	'''
	print('\n')
	print('Defining Encoder ...')

	input_ = Input(shape=(inp_size,))
	hidden_lin = Dense(nhids, kernel_initializer='normal')(input_)
	hidden_act = Activation(act_fn)(hidden_lin)
	output = Dense(inp_size)(hidden_act)
	AE = Model(input_,output)
	encoder = Model(input_,hidden_act) 

	if weights_path:
		AE.load_weights(weights_path)
	adam = Adam(lr=0.0005)
	AE.compile(loss=wrapper_loss(hiddens=hidden_act,lamda=lamda,decoder=False), optimizer=adam)

	return(AE,encoder)


## Define the Logistic Classifier in Keras
def logistic_classifier(inp_size):
	''' This function returns a logistic classifer 
		Which is used for classifying the encoded features
		This classifier uses low learning rate. 

		Arguments:
			inp_size - No. of input features for the logistic classifier

		Returns:
			Logistic Classifier model
	'''
	adam = Adam(lr=0.00005)
	model_cls = Sequential()
	model_cls.add(Dense(n_classes,input_shape=(inp_size,),activation='softmax')) #
	model_cls.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])

	return model_cls


## Function to train the OVR-Encoder and classify its encodings
def run(nhids, lamda, act_fn, ae_epochs,bs):
	''' This function
		1. Trains the Encoder model, also saves the model checkpoint at the least validation loss during training 
		2. Load the weights corresponding to the best checkpoint
		3. Gets encodings of cifar-10 data using the encoder
		4. Trains a logistic classifier on the encodings

		Arguments:
			nhids - Hidden layer size (encoder size) in Encoder/AutoEncoder
			lamda - Regularization strength for OVR Cost
			act_fn - Activation function of the encoder
			ae_epochs - Epochs for the encoder/AutoEncoder
			bs - Batch Size for the encoder/AutoEncoder

		Returns:
			Sparsity of the representations from encoder
			Accuracy of logistic classifier on the representations from encoder

		Note: both sparsity & accuracy are computed on test data
	'''

	##############
	tag = 'ovronly_ae_keras'+'_act_'+act_fn+'_nh_'+str(nhids)+'_lamda_'+str(lamda)+'_ae_epochs_'+str(ae_epochs)+'_bs_'+str(bs)
	##############
	print (tag)

	ae_wts = directory+'/'+tag+'_weights.h5'
	chkp = ModelCheckpoint(filepath=ae_wts,monitor='val_loss',period=1,save_best_only=True,save_weights_only=False,verbose=0)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', cooldown=0, min_lr=0.000001)		
	cl = [reduce_lr, chkp]

	print('\n')
	print('Training the Encoder model with OVR Loss ...')
	AE, encoder = autoencoder(inp_size=inp_size,nhids=nhids,act_fn=act_fn,lamda=lamda)
	AE.fit(X,X,validation_data=(test_X,test_X),epochs=ae_epochs, batch_size=bs, callbacks=cl, verbose=1)

	print('Loading the best weights for Encoder model ...')
	if os.path.isfile(ae_wts):
		AE, encoder = autoencoder(inp_size=inp_size,nhids=nhids,act_fn=act_fn,lamda=lamda,weights_path=ae_wts)

	print('Getting encodings for train & test data ...')
	train_hids = encoder.predict(X)
	test_hids = encoder.predict(test_X)

	# print('Saving the sparsity histogram plot')
	# plt.clf()
	# plt.hist(test_hids.reshape(-1), bins=40)
	# plt.savefig(directory+tag+'_hist.jpg')

	print('\n')
	print('Training Logistic model on the encodings ...')
	log_reg = logistic_classifier(inp_size=nhids)
	hist = log_reg.fit(train_hids,Y,validation_data=[test_hids,test_Y],callbacks=[reduce_lr], epochs=100, verbose=1).history

	r1 = test_hids[test_hids<0.1].shape[0]/float(test_hids.size)
	r2 = np.max(hist['val_acc'])

	return(r1,r2)


## Main function
if __name__== '__main__':
	''' This is the main function
		1. Parses the arguments from command line
		2. Loads PCA reduced Cifar-10 data. 
		3. Creates a directory for saving models, plots etc.
		4. Calls the 'run' function. Refer to this function for more details.
	'''

	args = parser.parse_args(sys.argv[1:])

	global directory, n_classes
	n_comp = args.n_comp
	nhids = args.encoded_size
	lamda = args.lamda
	act_fn = args.act_fn
	ae_epochs = args.encoder_epochs
	bs = args.batch_size
	directory = args.model_path
	
	## Load PCA reduced cifar-10 data
	X,Y,test_X,test_Y = load_cifar10_pca(n_comp=n_comp, scale=True)
	print('\n')
	print('Final data')
	print(X.shape)
	print(Y.shape)
	print(test_X.shape)
	print(test_Y.shape)

	inp_size = X.shape[1]
	n_classes = Y.shape[1]

	## Directory to save the models, plots etc.
	if directory not in os.listdir('.'):
		os.mkdir(directory)

	## Train the network
	tf.reset_default_graph()
	with tf.Session() as sess:
		r1, r2 = run(nhids=nhids, lamda=lamda, act_fn=act_fn, ae_epochs=ae_epochs, bs=bs)
		print('Accuracy -- {} | Sparsity -- {}'.format(r2,r1))
