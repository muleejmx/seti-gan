from pylab import *
import os
import numpy as np
import pandas as pd
import csv
import pickle
import fitsio

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling

import matplotlib.pyplot as plt
from scipy.misc import imread


print("Hello world!")
print(keras.__version__)

def one_hot(i):
	a = np.zeros(5, np.int32)
	a[i] = 1
	return a

# stop randomness
seed = 128
rng = np.random.RandomState(seed)

# set path
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

# dictionary of data: labels
with open('labelled_files.pkl', 'rb') as f:
    data = pickle.load(f)

data = [k for k, v in data.iteritems() if v == 4.0]

TRAIN_LIM = int(len(data) * 0.8)

train_x = []
train_y = []

for d in data[:TRAIN_LIM]:
	train_x.append(fitsio.read('/Users/mulan/desktop/fits_data/' + d +  '.fits'))
	train_y.append(one_hot(4))

train_x = np.asarray(train_x)
#define vars
g_input_shape = 100
d_input_shape = (16, 512)
hidden_1_num_units = 500
hidden_2_num_units = 500
g_output_num_units = 8192
d_output_num_units = 1
epochs = 25
batch_size = 128


# generator
model_1 = Sequential([
    Dense(units=hidden_1_num_units,
    	input_dim=g_input_shape,
    	activation='relu',
    	kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units,
    	activation='relu', 
    	kernel_regularizer=L1L2(1e-5, 1e-5)),
        
    Dense(units=g_output_num_units,
    	activation='sigmoid',
    	kernel_regularizer=L1L2(1e-5, 1e-5)),
    
    Reshape(d_input_shape),
])

# discriminator
model_2 = Sequential([
    InputLayer(input_shape=d_input_shape),
    
    Flatten(),
        
    Dense(units=hidden_1_num_units,
    	activation='relu',
    	kernel_regularizer=L1L2(1e-5, 1e-5)),

    Dense(units=hidden_2_num_units,
    	activation='relu',
    	kernel_regularizer=L1L2(1e-5, 1e-5)),
        
    Dense(units=d_output_num_units,
    	activation='sigmoid',
    	kernel_regularizer=L1L2(1e-5, 1e-5)),
])

gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))

model = AdversarialModel(base_model=gan,
	player_params=[model_1.trainable_weights, model_2.trainable_weights])

model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
	player_optimizers=['adam', 'adam'],
	loss='binary_crossentropy')

history = model.fit(x=train_x,
	y=gan_targets(train_x.shape[0]),
	epochs=100,
	batch_size=batch_size)


# plt.plot(history.history['player_0_loss'])
# plt.plot(history.history['player_1_loss'])
# plt.plot(history.history['loss'])

# plt.show()

zsamples = np.random.normal(size=(10, 100))

pred = model_1.predict(zsamples)

for i in range(pred.shape[0]):
    plt.imshow(pred[i, :], cmap='gray', aspect = 'auto')
    plt.show()
# img = fitsio.read('/Users/mulan/desktop/fits_data/' + data[0] + '.fits', flatten=True)

# plt.imshow(img, cmap='gray', aspect = 'auto')
# plt.axis('off')
# plt.show()








