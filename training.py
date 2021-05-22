import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import winsound
from contextlib import redirect_stdout


img_dims = (96, 96, 3)
data = []
labels = []


# set callback class to avoid overfitting
class StopTraining(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('accuracy') > 0.98:
            print("\n\n*************************************************************************************")
            print("\nReached 98% accuracy so, cancelling the training !!!")
            print("\n\n*************************************************************************************")
            self.model.stop_training = True
            winsound.Beep(frequency = 2500, duration = 1500)


stop_training = StopTraining()

# load image
image_files = [f for f in glob.glob(r'F:\Study\CLG\Project\Code\gender_dataset_face' + "/**/*", recursive = True)
               if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    data.append(image)
    label = img.split(os.path.sep)[-2]
    # F:\Study\CLG\Project\Code\dataset\Female\face_452.jpg
    if label == "Female":
        label = 1
    else:
        label = 0
    labels.append([label])
    # [[1],[0],[0],...]

# preprocessing data
data = np.array(data, dtype = "float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)
trainY = to_categorical(trainY, num_classes = 2)
# [[1,0], [0,1], [0,1],...]
testY = to_categorical(testY, num_classes = 2)

# Image Data Generator
aug = ImageDataGenerator(rotation_range = 25,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         shear_range = 0.2,
                         zoom_range = 0.2,
                         horizontal_flip = True,
                         fill_mode = "nearest")

# building model
classes = 2
inputShape = (img_dims[1], img_dims[0], img_dims[2])
chanDim = -1
if k.image_data_format() == "channels_first":
    inputShape = (img_dims[1], img_dims[0], img_dims[2])
    chanDim = 1

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', padding = "same", input_shape = inputShape),
    tf.keras.layers.BatchNormalization(axis = chanDim),
    tf.keras.layers.MaxPooling2D(pool_size = (3, 3)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.BatchNormalization(axis = chanDim),

    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.BatchNormalization(axis = chanDim),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.BatchNormalization(axis = chanDim),

    tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', padding = "same"),
    tf.keras.layers.BatchNormalization(axis = chanDim),
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Activation("relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(classes),
    tf.keras.layers.Activation("sigmoid")
    ])

with open('Model Summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        print("\n\n*************************************************************************************")
        print("\n\nModel Summary file created !\n\n")
        print("\n\n*************************************************************************************")


# compile
opt = Adam(lr = 1e-3, decay = 1e-3 / 100)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

epochs = 100

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = 64),
                        validation_data = (testX, testY),
                        steps_per_epoch = len(trainX)//64,
                        epochs = epochs,
                        verbose = 1,
                        callbacks = [stop_training])

# save
model.save('gender_detection.model')
print("\n\n*************************************************************************************")
print("\n\nModel created and save to current working directory !\n\n")
print("\n\n*************************************************************************************")


# plot graph
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label = "training loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "validation loss")
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc = "upper right")
plt.savefig("Training Loss vs Validation Loss.png")
plt.show()


plt.plot(np.arange(0, N), H.history["accuracy"], label = "training accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label = "validation accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc = "upper right")
plt.savefig("Training Accuracy vs Validation Accuracy.png")
plt.show()

print("Model successfully build !!")
winsound.Beep(frequency = 2500, duration = 1500)
print("\n\n*************************************************************************************")
