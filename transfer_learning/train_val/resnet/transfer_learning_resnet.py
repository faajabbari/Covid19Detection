import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np
import glob

import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory


# ## Parameters


parser = argparse.ArgumentParser()
parser.add_argument('-p',
                    help='path to dataset')

args = parser.parse_args()

BATCH_SIZE = 32
IMG_SIZE = (512, 512)

initial_epochs = 15
fine_tune_epochs = 100
name = ''

# ## Data preprocessing


PATH = args.p

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

#import pudb; pu.db
validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)


test_dataset = image_dataset_from_directory(test_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
class_names = train_dataset.class_names
print(class_names)

# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")

# Class weight
'''files_per_class = []
for folder in os.listdir(train_dir):
    if not os.path.isfile(folder):
            files_per_class.append(len(os.listdir(train_dir + '/' + folder)))
total_files = sum(files_per_class)
class_weights = {}
for i in range(len(files_per_class)):
    # class_weights[i] = 1 - (float(files_per_class[i]) / total_files)
    class_weights[i] = (float(files_per_class[i]) / total_files)
print (class_weights)
'''

# ### Configure the dataset for performance


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ### Use data augmentation


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)
])


preprocess_input = tf.keras.applications.resnet.preprocess_input


# Create the base model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# Feature extraction
base_model.trainable = False
base_model.summary()


# Add a classification head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)



prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


inputs = tf.keras.Input(shape=(512, 512, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
# import pudb; pu.db
len(model.trainable_variables)


early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
check_point = ModelCheckpoint("./my_model_resnet"+name+".h5", monitor="val_loss", save_best_only=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6)

callbacks_list = [early, check_point, reduce]
# callbacks_list = [check_point, reduce]

# Train the model
history = model.fit(train_dataset,
        epochs=initial_epochs,
        validation_data=validation_dataset,
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
plt.savefig('pretrained_model_results_w3_lr2'+name+'.png')

# Fine Tuning
# ### Un-freeze the top layers of the model

base_model.trainable = True


print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 120

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
check_point2 = ModelCheckpoint("./my_model_resnet_fine_tune_w3"+name+".h5", monitor="val_loss", save_best_only=True)
#reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6)

callbacks_list2 = [early, check_point2]#, reduce]
callbacks_list2 = [check_point2]#, reduce]
import pudb; pu.db
history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
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
plt.savefig('fine_tuned_model_results_w3_lr2'+name+'.png')


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
  plt.savefig('reesnet'+name+'.png')
