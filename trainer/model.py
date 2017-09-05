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

data_loc = '/Users/mulan/desktop/fits_data/'
ext = '.fits'

typ = '_diag'
# stop randomness
seed = 128
rng = np.random.RandomState(seed)

# set path
root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

# dictionary of data: labels
with open('labelled_files.pkl', 'rb') as f:
    data = pickle.load(f)

data = [k for k, v in data.iteritems() if v == 6.0 or v == 7.0]

fig = plt.figure()
plt.imshow(fitsio.read(data_loc + str(data[0]) + ext), cmap='gray', aspect = 'auto')
fig.savefig('orig.png', dpi = fig.dpi, transparent = True)


# def flatten(x):
#     return [item for sublist in x for item in sublist]

# pix_data = np.array(flatten([fitsio.read(data_loc + d + ext) for d in data]))

# binary data
with open('binary_diag.pkl', 'rb') as f:
    data = pickle.load(f)

fig = plt.figure()
plt.imshow(data[0], cmap='gray', aspect = 'auto')
fig.savefig('binary.png', dpi = fig.dpi, transparent = True)


# thresholding code
# threshold = np.sort(pix_data.flatten())
# print(len(threshold))

# plt.hist(threshold, bins = 100, range = [0, 0.78*10**9])
# plt.show()

# THRESH = threshold[(int(len(threshold) * 0.89))]
# binary_data = np.copy(pix_data)
# binary_data[binary_data < THRESH] = 0
# binary_data[binary_data >= THRESH] = 1
# binary_data = np.reshape(binary_data, (len(data), 16, 512))

# with open('binary_diag.pkl', 'wb') as outfile:
#     pickle.dump(binary_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# plt.imshow(binary_data[1], aspect = 'auto', cmap = 'gray')
# plt.axis('off')
# plt.show()
# plt.imshow(fitsio.read(data_loc + data[1] + ext), cmap='gray', aspect = 'auto')
# plt.axis('off')
# plt.show()


# TRAIN_LIM = int(len(data) * 0.8)

# # train_x = []

# # for d in data[:TRAIN_LIM]:
# # 	train_x.append(fitsio.read('/Users/mulan/desktop/fits_data/' + d +  '.fits'))

# # train_x = np.asarray(train_x)
# train_x = data[:TRAIN_LIM]
# print(train_x.shape)

# #define vars
# g_input_shape = 100
# d_input_shape = (16, 512)
# hidden_1_num_units = 500
# hidden_2_num_units = 500
# g_output_num_units = 8192
# d_output_num_units = 1
# epochs = 200
# batch_size = 128


# # generator
# model_1 = Sequential([
#     Dense(units=hidden_1_num_units,
#     	input_dim=g_input_shape,
#     	activation='relu',
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),

#     Dense(units=hidden_2_num_units,
#     	activation='relu', 
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),
        
#     Dense(units=g_output_num_units,
#     	activation='sigmoid',
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),
    
#     Reshape(d_input_shape),
# ])

# # discriminator
# model_2 = Sequential([
#     InputLayer(input_shape=d_input_shape),
    
#     Flatten(),
        
#     Dense(units=hidden_1_num_units,
#     	activation='relu',
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),

#     Dense(units=hidden_2_num_units,
#     	activation='relu',
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),
        
#     Dense(units=d_output_num_units,
#     	activation='sigmoid',
#     	kernel_regularizer=L1L2(1e-5, 1e-5)),
# ])

# gan = simple_gan(model_1, model_2, normal_latent_sampling((100,)))

# model = AdversarialModel(base_model=gan,
# 	player_params=[model_1.trainable_weights, model_2.trainable_weights])

# model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
# 	player_optimizers=['adam', 'adam'],
# 	loss='binary_crossentropy')

# history = model.fit(x=train_x,
# 	y=gan_targets(train_x.shape[0]),
# 	epochs=epochs,
# 	batch_size=batch_size)

# fig = plt.figure()

# plt.plot(history.history['player_0_loss'])
# plt.plot(history.history['player_1_loss'])
# plt.plot(history.history['loss'])

# fig.savefig('losses.png', dpi = fig.dpi)

# zsamples = np.random.normal(size=(10, 100))

# pred = model_1.predict(zsamples)

# for i in range(pred.shape[0]):
# 	fig = plt.figure()
# 	plt.imshow(pred[i, :], cmap='gray', aspect = 'auto')
# 	fig.savefig(str(i) + typ + '_' + str(epochs) + '.png', dpi = fig.dpi, transparent = True)
# img = fitsio.read('/Users/mulan/desktop/fits_data/' + data[0] + '.fits', flatten=True)

# plt.imshow(img, cmap='gray', aspect = 'auto')
# plt.axis('off')
# plt.show()








