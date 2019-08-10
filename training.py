import pandas as pd
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from time import time
from keras.callbacks import TensorBoard
import math


runningN = "ResNet50_"+str(time())
# path_train = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data/train'
path_train = '/media/DISK8gb/cropped/car_data/train'
# path_test = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data/train'
path_test = '/media/DISK8gb/cropped/car_data/test'
epochs = 1000
batch_size = 8


# imports the mobilenet model and discards the last 1000 neuron layer.
base_model = MobileNet(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', name="dense1")(x)
x = Dense(1024, activation='relu', name="dense2")(x)
x = Dense(512, activation='relu', name="dense3")(x)
preds = Dense(196, activation='softmax', name="dense4-c")(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers:
    layer.trainable = True

model.summary()


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory(path_train,  # this is where you specify the path to the main data folder
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
test_generator = train_datagen.flow_from_directory(path_test,  # this is where you specify the path to the main data folder
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=True)

tfboard = TensorBoard(log_dir="logs/{}".format(), histogram_freq=1,
                      batch_size=batch_size, write_images=True, write_graph=True, write_grads=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

step_size_train = train_generator.n//train_generator.batch_size
v_steps = math.ceil(test_generator.n / batch_size)
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=v_steps,
                    callbacks=[tfboard]
                    )
test_loss, test_acc = model.evaluate_generator(
    generator=train_generator, steps=24)
print(test_loss)
print(test_acc)
model.save(runningN+".h5")
