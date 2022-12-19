import os
import sys

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

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
#import pudb; pu.db

h = 64
w = 64
d=20
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
#    import pudb; pu.db
    scan = nib.load(filepath) #.tolist())
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
    desired_depth = d
    desired_width = h
    desired_height = w
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
   # img = tf.expand_dims(img, axis=3)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
#    import pudb; pu.db
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

random.seed(778)
main_dir = sys.argv[1]
# import pudb; pu.db
normal_scan_paths = [os.path.join(main_dir, 'CT_0', x) for x in os.listdir(os.path.join(main_dir, 'CT_0'))]
abnormal_scan_paths1 = [os.path.join(main_dir, 'CT_1', x) for x in os.listdir(os.path.join(main_dir, 'CT_1'))]
abnormal_scan_paths2 = [os.path.join(main_dir, 'CT_2', x) for x in os.listdir(os.path.join(main_dir, 'CT_2'))]
abnormal_scan_paths3 = [os.path.join(main_dir, 'CT_3', x) for x in os.listdir(os.path.join(main_dir, 'CT_3'))]
abnormal_scan_paths4 = [os.path.join(main_dir, 'CT_4', x) for x in os.listdir(os.path.join(main_dir, 'CT_4'))]
n = len(normal_scan_paths) - len(abnormal_scan_paths2) - len(abnormal_scan_paths3)
print('random len = ' + str(n))
abnormal_scan_paths = abnormal_scan_paths2  + abnormal_scan_paths3 + random.sample(abnormal_scan_paths1, n)

# normal_scan_paths = normal_scan_paths[1:30]
# abnormal_scan_paths = abnormal_scan_paths[1:30]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

#import pudb; pu.db
list_IDs = normal_scan_paths + abnormal_scan_paths
list_IDs = list_IDs[100]
random.shuffle(list_IDs)
length = math.ceil(0.8 * len(list_IDs))
list_IDs_train = list_IDs[:length]
list_IDs_validation = list_IDs[length:]
 
print("______________________________________")
print("samples of train set: ", len(list_IDs_train))
print("samples of validation set: ", len(list_IDs_validation)) 


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(h, w, d),
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
        #import pudb; pu.db
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 1)) 
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#            import pudb; pudb
            if self.mode == 'train':
                raw_data = process_scan(ID)
                augumented_data = rotate(raw_data)
        #        if random.random() > 0:
         #           augumented_data = np.fliplr(augumented_data)
                X[i,] = tf.expand_dims(augumented_data, axis=3)  # augumented_data
            else:
                X[i,] = tf.expand_dims(process_scan(ID), axis=3)



            # Store class
#            import pudb; pu.db
            label = ID.split('/')[-2]
            if label == 'CT_0':
                y[i] = 0
#            elif label == 'CT_1':
#                y[i] = 1
#            elif label == 'CT_2':
#                y[i] = 2
#            elif label == 'ct_3':
#                y[i] = 3
            else:
                y[i] = 1

#        import pudb; pu.db        
#        y_hat = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y # y_hat # keras.utils.to_categorical(y, num_classes=self.n_classes)




# import pudb; pu.db
# ## Data augmentation

def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 0, 0, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


#def train_preprocessing(volume, label):
#    """Process training data by rotating and adding a channel."""
#    # Rotate volume
#    volume = rotate(volume)
#    volume = tf.expand_dims(volume, axis=3)
#    return volume, label

#def validation_preprocessing(volume, label):
#    """Process validation data by only adding a channel."""
#    volume = tf.expand_dims(volume, axis=3)
#    return volume, label


# import pudb; pu.db
training_generatorin = DataGenerator(list_IDs_train, mode='train')#tf.data.Dataset.from_generator(DataGenerator(list_IDs_train, mode='train'), (tf.float32, tf.int16))
validation_generatorin = DataGenerator(list_IDs_validation, mode='validation') # tf.data.Dataset.from_generator(DataGenerator(list_IDs_validation, mode='val'), (tf.float32, tf.int16)) 
                                                           
# Augment the on the fly during training.
#train_dataset = (
#    train_loader.shuffle(len(x_train))
#    .map(train_preprocessing)
#    .batch(batch_size)
#    .prefetch(2)
#)

#print('Val_dataset')
# Only rescale.
#validation_dataset = (
#    validation_loader.shuffle(len(x_val))
#    .map(validation_preprocessing)
#    .batch(batch_size)
#    .prefetch(2)
#)


# Visualize an augmented CT scan.
#data = train_dataset.take(1)
#images, labels = list(data)[0]
#images = images.numpy()
#image = images[0]
#print("Dimension of the CT scan is:", image.shape)
#plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
#plt.savefig('random_input.png')


#def plot_slices(num_rows, num_columns, width, height, data):
#    """Plot a montage of 20 CT slices"""
#    data = np.rot90(np.array(data))
#    data = np.transpose(data)
#    data = np.reshape(data, (num_rows, num_columns, width, height))
#    rows_data, columns_data = data.shape[0], data.shape[1]
#    heights = [slc[0].shape[0] for slc in data]
#    widths = [slc.shape[1] for slc in data[0]]
#    fig_width = 12.0
#    fig_height = fig_width * sum(heights) / sum(widths)
#    f, axarr = plt.subplots(
#        rows_data,
#        columns_data,
#        figsize=(fig_width, fig_height),
#        gridspec_kw={"height_ratios": heights},
#    )
#    for i in range(rows_data):
#        for j in range(columns_data):
#            axarr[i, j].imshow(data[i][j], cmap="gray")
#            axarr[i, j].axis("off")
#    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
#plot_slices(4, 10, 128, 128, image[:, :, :40])


# ## Define a 3D convolutional neural network
# 
# To make the model easier to understand, we structure it into blocks.
# The architecture of the 3D CNN used in this example
# is based on [this paper](https://arxiv.org/abs/2007.13224).

# In[ ]:



def get_model(width=h, height=w, depth=d):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
#    outputs = layers.Dense(units=5, activation="softmax")(x)
    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# ## Train model

# In[ ]:


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
#    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_image_classification_with_augumentationi_2c.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

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
    callbacks=[checkpoint_cb, early_stopping_cb],
)


model.save('caution.h5')

# It is important to note that the number of samples is very small (only 200) and we don't
# specify a random seed. As such, you can expect significant variance in the results. The full dataset
# which consists of over 1000 CT scans can be found [here](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1). Using the full
# dataset, an accuracy of 83% was achieved. A variability of 6-7% in the classification
# performance is observed in both cases.

# ## Visualizing model performance
# 
# Here the model accuracy and loss for the training and the validation sets are plotted.
# Since the validation set is class-balanced, accuracy provides an unbiased representation
# of the model's performance.

# In[ ]:


#fig, ax = plt.subplots(1, 2, figsize=(20, 3))
#ax = ax.ravel()

#for i, metric in enumerate(["acc", "loss"]):
#    ax[i].plot(model.history.history[metric])
#    ax[i].plot(model.history.history["val_" + metric])
#    ax[i].set_title("Model {}".format(metric))
#    ax[i].set_xlabel("epochs")
#    ax[i].set_ylabel(metric)
#    ax[i].legend(["train", "val"])


# ## Make predictions on a single CT scan

# In[ ]:


# Load best weights.
#model.load_weights("3d_image_classification.h5")
#prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
#scores = [1 - prediction[0], prediction[0]]
#
#class_names = ["normal", "abnormal"]
#for score, name in zip(scores, class_names):
#    print(
#        "This model is %.2f percent confident that CT scan is %s"
#        % ((100 * score), name)
#    )

