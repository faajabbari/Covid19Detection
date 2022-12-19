# -*- coding: utf-8 -*-
"""Copy of 3D_image_classification

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16NMqZc09TwCuWzDmxIfb4kdw7dB6GCxT
"""

import os
import sys

import zipfile
import numpy as np
import tensorflow as tf
import math
import tqdm
import zipfile
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import random
from scipy import ndimage
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence


random.seed(777)
main_dir = sys.argv[1]
main_dir2 = sys.argv[2]
#import pudb; pu.db
normal_scan_paths1 = [os.path.join(main_dir, 'CT_0', x) for x in os.listdir(os.path.join(main_dir, 'CT_0'))]
abnormal_scan_paths1 = [os.path.join(main_dir, 'CT_1', x) for x in os.listdir(os.path.join(main_dir, 'CT_1'))]
abnormal_scan_paths2 = [os.path.join(main_dir, 'CT_2', x) for x in os.listdir(os.path.join(main_dir, 'CT_2'))]
abnormal_scan_paths3 = [os.path.join(main_dir, 'CT_3', x) for x in os.listdir(os.path.join(main_dir, 'CT_3'))]
abnormal_scan_paths4 = [os.path.join(main_dir, 'CT_4', x) for x in os.listdir(os.path.join(main_dir, 'CT_4'))]
# n = len(normal_scan_paths) - len(abnormal_scan_paths2) - len(abnormal_scan_paths3)
# print('random len = ' + str(n))
# abnormal_scan_paths = abnormal_scan_paths2  + abnormal_scan_paths3 + random.sample(abnormal_scan_paths1, n)

normal_scan_dicom_paths = [os.path.join(main_dir2, 'normal', x) for x in os.listdir(os.path.join(main_dir2, 'normal'))]
abnormal_scan_dicom_paths = [os.path.join(main_dir2, 'covid', x) for x in os.listdir(os.path.join(main_dir2, 'covid'))]
# normal_scan_paths = normal_scan_paths[1:30]
# abnormal_scan_paths = abnormal_scan_paths[1:30]

normal_scan_paths = normal_scan_paths1 + normal_scan_dicom_paths

# n = len(normal_scan_paths) - len(abnormal_scan_paths2) - len(abnormal_scan_paths3) - len(abnormal_scan_dicom_paths)
# print('random len = ' + str(n))
abnormal_scan_paths = abnormal_scan_paths2  + abnormal_scan_paths3 + abnormal_scan_dicom_paths# + random.sample(abnormal_scan_paths1, n)

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))



iheight = 128
iwidth = 128
idepth = 64

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = idepth
    desired_width = iwidth
    desired_height = iheight
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

list_IDs = normal_scan_paths + abnormal_scan_paths
# import pudb; pu.db
random.shuffle(list_IDs)
length = math.ceil(0.8 * len(list_IDs))
list_IDs_train = list_IDs[:length]
list_IDs_validation = list_IDs[length:]

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=8, dim=(iheight, iwidth, idepth),
                 n_classes=2, shuffle=True, mode='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.on_epoch_end()
        self.mode = mode


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
#        import pudb; pu.db
        X, y = self.__data_generation(list_IDs_temp)


        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          if self.mode == 'train':
            import pudb; pu.db
            raw_data = process_scan(ID)
            augumented_data = aug(raw_data)
#            X[i,] = tf.expand_dims(augumented_data, axis=3)
            X[i,] = np.expand_dims(augumented_data, axis=3)
            # print(X[i,].shape)
          else:
            data = process_scan(ID)
#            data = tf.convert_to_tensor(data, dtype='float32')
#            X[i,] = tf.expand_dims(data, axis=3)
            data = np.array(data, dtype='float32')
            X[i,] = np.expand_dims(data, axis=3) 

            import pudb; pu.db
          label = ID.split('/')[-2]
          # print(label)
          if label == 'CT_0' or label == 'normal':
                y[i] = 0
                #import pudb; pu.db

          else:
                y[i] = 1
               # import pudb; pu.db
        X = tf.convert_to_tensor(X)
        return X, y # keras.utils.to_categorical(y, num_classes=self.n_classes)

#@tf.function
def aug(volume, numTrans=1):
    import pudb; pu.db
    whichTrans  = np.random.choice([0, 1, 2, 3], numTrans, replace=False)
    if 0 in whichTrans:
        def func(volume):
            return volume
    if 1 in whichTrans:


            """Rotate the volume by a few degrees"""

            def func(volume):
             # define some rotation angles
                angles = [-20, -10, -5, 5, 10, 20]
                # pick angles at random
                angle = random.choice(angles)
                # rotate volume
                volume = ndimage.rotate(volume, angle, reshape=False)
                volume[volume < 0] = 0
                volume[volume > 1] = 1
  #              volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
                return volume
    if 2 in whichTrans:
            def func(volume, isseg=False):
                offsetx = np.random.choice([0, 1, 2, 3, 4])
                offsety = np.random.choice([0, 1, 2, 3, 4])
                offset = [offsetx, offsety]
                order = 0 if isseg == True else 5

                volume = ndimage.interpolation.shift(volume, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')
 #               volume = tf.numpy_function(translateit, [volume], tf.float32)
                return volume

    if 3 in whichTrans:
            def func(volume):
                image = np.fliplr(volume)
#                volume = tf.numpy_function(flipit, [volume], tf.float32)
                return volume

        #elif 4 in WhichTrans:



    augmented_volume = func(np.array(volume, dtype='float32')) # tf.numpy_function(func, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=iwidth, height=iheight, depth=idepth):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((iwidth, iheight, idepth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

#    x = ilayers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
#    x = layers.MaxPool3D(pool_size=2)(x)
#    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

#    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
#    x = layers.MaxPool3D(pool_size=2)(x)
#    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
#import pudb; pu.db
model = get_model(width=iwidth, height=iheight, depth=idepth)
model.summary()

# Compile model.
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification_lr0_001_b8_dropout0_5.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)
tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)

"""## Train model"""

training_generatorin = DataGenerator(list_IDs_train, mode='train')#tf.data.Dataset.from_generator(DataGenerator(list_IDs_train, mode='train'), (tf.float32, tf.int16))
validation_generatorin = DataGenerator(list_IDs_validation, mode='validation')

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit_generator(generator=training_generatorin,
     validation_data=validation_generatorin,
#    train_dataset,
#    validation_data=validation_dataset,
     epochs=epochs,
     shuffle=True,
     verbose=1,
    #class_weight=class_weights,
     callbacks=[checkpoint_cb, early_stopping_cb, tb_callback]
)

"""It is important to note that the number of samples is very small (only 200) and we don't
specify a random seed. As such, you can expect significant variance in the results. The full dataset
which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
performance is observed in both cases.

## Visualizing model performance

Here the model accuracy and loss for the training and the validation sets are plotted.
Since the validation set is class-balanced, accuracy provides an unbiased representation
of the model's performance.
"""

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

plt.savefig('3d_res_less.png')
"""## Make predictions on a single CT scan"""

# Commented out IPython magic to ensure Python compatibility.
# Load best weights.
model.load_weights("3d_image_classification_less_complex.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
#         % ((100 * score), name)
    )
