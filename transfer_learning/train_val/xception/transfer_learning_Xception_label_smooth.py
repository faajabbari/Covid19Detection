import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
import glob
import cv2

import  tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import Sequence

# ## Parameters


parser = argparse.ArgumentParser()
parser.add_argument('-p',
                    help='path to dataset')

args = parser.parse_args()

BATCH_SIZE = 32
IMG_SIZE = (512, 512)

initial_epochs = 15
fine_tune_epochs = 100
name = 'without_window_20_30'

# ## Data preprocessing


PATH = args.p

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    #import pudb; pu.db
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(512,512,3), n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #import pudb; pu.db
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #import pudb; pu.db
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def smooth_labels(self, labels, factor=0.1):
        # smooth the labels
        labels = np.array(labels, dtype='float')
        labels *= (1 - factor)
        labels += (factor / len(labels))
        # returned the smoothed labels
        return labels

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #import pudb; pu.db
            X[i,] = cv2.imread(ID)
            label = ID.split('/')[-2]
            if label == 'covid':
                Y[i] = 1

            else:
                Y[i] = 0

        yy = keras.utils.to_categorical(Y, num_classes=2)
        y = self.smooth_labels(yy)
        
        return np.asarray(X), np.asarray(y) #
        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


#params = {'dim': (32,32,32),
#          'batch_size': 64,
#          'n_classes': 6,
#          'n_channels': 1,
#          'shuffle': True}

# Datasets
train_dataset = glob.glob(train_dir + '/*/*')
val_dataset = glob.glob(validation_dir + '/*/*')
# Generators
training_generator = DataGenerator(train_dataset)
validation_generator = DataGenerator(val_dataset)
                                     

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)
])


preprocess_input = tf.keras.applications.xception.preprocess_input
#import pudb;pu.db

# Create the base model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

# Feature extraction
base_model.trainable = False
base_model.summary()


# Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(2)


inputs = tf.keras.Input(shape=(512, 512, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.1
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# import pudb; pu.db
len(model.trainable_variables)


early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
check_point = ModelCheckpoint("./my_model_xception_smooth_label"+name+".h5", monitor="val_loss", save_best_only=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6)

callbacks_list = [early, check_point, reduce]
# callbacks_list = [check_point, reduce]

# Train the model
#import pudb; pu.db
history = model.fit_generator(training_generator,
        epochs=initial_epochs,
        validation_data=validation_generator,
        #class_weight=class_weights,
        callbacks=callbacks_list)


# ### Learning curves
#model.save('whit_out_fine_tune.h5')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('pretrained_model_results_w3_lr2_smooth_label_xception'+name+'.png')

# Fine Tuning
# ### Un-freeze the top layers of the model

base_model.trainable = True


print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 150

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# ### Compile the model

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/300),
              metrics=['accuracy'])

model.summary()

len(model.trainable_variables)


# ### Continue training the model


total_epochs =  initial_epochs + fine_tune_epochs

early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
check_point2 = ModelCheckpoint("./my_model_xception_fine_tune_smooth_label"+name+".h5", monitor="val_loss", save_best_only=True)
#reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6)

callbacks_list2 = [early, check_point2]#, reduce]
callbacks_list2 = [check_point2]#, reduce]
#import pudb; pu.db
history_fine = model.fit_generator(training_generator,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_generator,
                        #  class_weight=class_weights,
                         callbacks=callbacks_list2)
# model.save('res_net.h5')


# ### Learning curves


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('fine_tuned_model_results_w3_lr2_smooth_label_xception'+name+'.png')


# ### Evaluation and prediction


loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


#Retrieve a batch of images from the test set
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
  plt.savefig('smooth_label_xception'+name+'.png')
