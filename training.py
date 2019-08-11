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
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from time import time
from keras.callbacks import TensorBoard
import math

models = {
    "resnet50": ResNet50(weights='imagenet', include_top=False),
    "vgg19": VGG19(weights='imagenet', include_top=False),
    "mobilenet": MobileNet(weights='imagenet', include_top=False),
    "mobilenetv2": MobileNetV2(weights='imagenet', include_top=False)
}


class DeepLearning_training():

    def __init__(self, runningN, path_train, path_test, epochs, batch_size, model):
        self.runningN = runningN+str(time())
        self.path_train = path_train
        self.path_test = path_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_name = model

        self._defineGenerator()
        self._defineModel()
        self._defineBoard()
        

    def _defineModel(self):
        base_model = models[self.model_name]
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu', name="dense1")(x)
        x = Dense(1536, activation='relu', name="dense2")(x)
        x = Dense(1024, activation='relu', name="dense3")(x)
        preds = Dense(196, activation='softmax', name="dense4-c")(x)

        self.model = Model(inputs=base_model.input, outputs=preds)

        for layer in self.model.layers[:-4]:
            layer.trainable = False
        for layer in self.model.layers[-4:]:
            layer.trainable = True

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def _defineGenerator(self):
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)  

        self.train_generator = self.train_datagen.flow_from_directory(self.path_train,  
                                                                      target_size=(
                                                                          224, 224),
                                                                      color_mode='rgb',
                                                                      batch_size=self.batch_size,
                                                                      class_mode='categorical',
                                                                      shuffle=True)
        self.test_generator = self.train_datagen.flow_from_directory(self.path_test,  
                                                                     target_size=(
                                                                         224, 224),
                                                                     color_mode='rgb',
                                                                     batch_size=self.batch_size,
                                                                     class_mode='categorical',
                                                                     shuffle=True)

    def _defineBoard(self):

        self.tfboard = TensorBoard(log_dir="logs/{}".format(self.runningN), histogram_freq=0,
                                   batch_size=self.batch_size, write_images=True, write_graph=True)

    def run_training(self):

        self.step_size_train = self.train_generator.n//self.train_generator.batch_size
        self.v_steps = math.ceil(self.test_generator.n / self.batch_size)
        self.model.fit_generator(generator=self.train_generator,
                                 steps_per_epoch=self.step_size_train,
                                 epochs=self.epochs,
                                 validation_data=self.test_generator,
                                 validation_steps=self.v_steps,
                                 callbacks=[self.tfboard]
                                 )
        test_loss, test_acc = self.model.evaluate_generator(
            generator=self.train_generator, steps=24)
        print(test_loss)
        print(test_acc)
        self.model.save(self.runningN+".h5")

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epochs), self.model.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), self.model.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), self.model.history["acc"], label="train_acc")
        plt.plot(np.arange(0, self.epochs), self.model.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")
