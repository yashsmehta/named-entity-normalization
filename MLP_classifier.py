import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import csv
import re
import pickle
import time
from datetime import timedelta
import pandas as pd

import utils

start = time.time()

inp_dir, seed = utils.parse_args()
n_classes = 2
write_file = True

np.random.seed(seed)
tf.random.set_seed(seed)

hidden_dim = 768

#modify this to load embeddings for all entity classes!
file = open(inp_dir + entity_class + '-' + 'embedings.pkl', 'rb')

data = pickle.load(file)
data_x, data_y = list(zip(*data))
file.close()

alphaW = np.zeros([12])
alphaW[11] = 1

inputs = []
targets = []

n_batches = len(data_y)

for ii in range(n_batches):
    inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
    targets.extend(data_y[ii])

inputs = np.array(inputs)
targets = np.array(targets)
batch_size = 32
lr = 1e-3
epochs = 5

targets = tf.keras.utils.to_categorical(targets, num_classes=n_classes)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(50, input_dim=hidden_dim, activation='relu'))
model.add(tf.keras.layers.Dense(n_classes))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['mse', 'accuracy'])

print(model.summary())

history = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2)

print('acc: ', history.history['accuracy'])
print('val acc: ', history.history['val_accuracy'])

print(timedelta(seconds=int(time.time() - start)), end=' ')

model.save('models')

if(write_file):
    results_file = 'mlp_results.csv'
    meta_info = (lr, epochs, seed)
    utils.file_writer(results_file, meta_info, history.history['val_accuracy'], history.history['val_loss'])

