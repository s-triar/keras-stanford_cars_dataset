# import pandas as pd
# import numpy as np
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import keras
# import matplotlib.pyplot as plt
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.applications.resnet50 import ResNet50
# from keras.applications.vgg19 import VGG19
# from keras.applications.mobilenet import MobileNet
# from keras.applications.mobilenetv2 import MobileNetV2
# from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model
# from keras.optimizers import Adam

# from time import time
# from keras.callbacks import TensorBoard
# import math
from keras.models import load_model

# models = {
#     "resnet50": ResNet50(weights='imagenet', include_top=False),
#     "vgg19": VGG19(weights='imagenet', include_top=False),
#     "mobilenet": MobileNet(weights='imagenet', include_top=False),
#     "mobilenetv2": MobileNetV2(weights='imagenet', include_top=False)
# }




class DeepLearning_testing():

    def __init__(self, path_model, path_data_test, batch_size):
        self.batch_size=batch_size
        self.path_test = path_data_test
        self.model = load_model(path_model)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self._defineGenerator()

    def _defineGenerator(self):
        self.train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)  # included in our dependencies

        self.test_generator = self.train_datagen.flow_from_directory(self.path_test,  # this is where you specify the path to the main data folder
                                                                        target_size=(
                                                                            224, 224),
                                                                        color_mode='rgb',
                                                                        batch_size=self.batch_size,
                                                                        class_mode='categorical',
                                                                        shuffle=True)
    
    def predict(self):
        filenames = self.test_generator.filenames
        nb_samples = len(filenames)
        return self.model.predict_generator(self.test_generator,steps = nb_samples)
        
        
    