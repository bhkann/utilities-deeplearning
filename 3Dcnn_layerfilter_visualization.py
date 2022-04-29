### VISUALIZATIONS ###
import os
import sys
import numpy as np
import pandas as pd
#
import tensorflow as tf

from keras import models
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns

import math

### LOAD YOUR PRETRAINED TF/KERAS MODEL ###
model_name = '{INSERT PATH OF MODEL}.h5'
model = load_model(model_name)

### LOAD YOUR DATA ###
### Should be a single input example (this code is written for 3D with single channel, but can adapt to 2D) e.g. (32,32,32,1)
### "image" : 3D numpy array input e.g. (32,32,32,1)

### Visualizing Layers ###
image = image.reshape(1,*image.shape)

'''
## To Plot single layer ##
layer_outputs = [layer.output for layer in model.layers[:15]] # CHOOSE NUMBER OF LAYERS , 15 is example; Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(image,1) # Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[11]
print(first_layer_activation.shape)
### Find center slice ###
midslice = first_layer_activation.shape[1]//2 
plt.matshow(first_layer_activation[0, midslice, :, :,32], cmap='viridis')
plt.show()
'''

## Loop through multiple layers ##
layer_outputs = [layer.output for layer in model.layers[:74]] # Extracts the outputs of the top 12 layers
layer_names = []
for layer in model.layers[0:74]: # choose the layers you wish to visualize
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict([image,image_small],1) # Returns a list of Numpy arrays: one array per layer activation

images_per_row = 16 # Choose how many images to display per row
#​
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    midslice = layer_activation.shape[1]//2 ## THIS PLOTS ACTIVATIONS AT THE MIDDLE SLICE, can modify as desired
    print('midslice: ', midslice) 
    if 'conv' in layer_name: # == 'last_64s':
        ### TO PLOT INTERMEDIARY LAYER CHANNELS ###
        try:
            print(layer_name, " shape: ", layer_activation.shape)
            size = layer_activation.shape[2] #Adjust if 2D vs 3D The feature map has shape (1, midslice, size, size, n_features).
            n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
            display_grid = np.zeros((size * n_cols, images_per_row * size))
            for col in range(n_cols): # Tiles each filter into a big horizontal grid
                for row in range(images_per_row):
                    try:
                        channel_image = layer_activation[0, midslice, :, :, col * images_per_row + row]
                        channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        display_grid[col * size : (col + 1) * size, # Displays the grid
                                     row * size : (row + 1) * size] = channel_image
                    except Exception as e:
                        print(e)
            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig('/{INSERT PATH}/Layers_' + layer_name + '.png', facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None, dpi=600)
        except Exception as e:
            print(e)

plt.show()


# TO VISUALIZE FILTERS/WEIGHTS
### Visualizing Features/Filters ####

# summarize filter shapes
for layer in model.layers:
    # check for convolutional layer
    #if 'conv' not in layer.name:
    if layer.name != 'conv1': # Or choose a layer
        continue
    filters, biases = layer.get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    print(layer.name, filters.shape)
    # load the model
    # plot first few filters
    n_filters, ix = 63, 1
    #plt.figure(figsize=(8,20))
    #plt.subplots_adjust(hspace=0.5)
    
    for i in range(n_filters):
        # get the filter
        f = filters[:, :, :, :, i]
        # plot each channel separately
        for j in range(3):
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters//3, 9, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.figure(figsize=(8, 6), dpi=80)
            # plot filter channel in grayscale
            plt.imshow(f[:, :, j], cmap='viridis')
            ix += 1
    # show the figure
    plt.suptitle("Filters for XXXX", fontsize=18, y=0.95)
    plt.savefig('/{INSERT PATH}', facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None, dpi=600)
    plt.show()