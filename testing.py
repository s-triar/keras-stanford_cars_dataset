from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

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
            preprocessing_function=preprocess_input)  

        self.test_generator = self.train_datagen.flow_from_directory(self.path_test, 
                                                                        target_size=(
                                                                            224, 224),
                                                                        color_mode='rgb',
                                                                        batch_size=self.batch_size,
                                                                        class_mode='categorical',
                                                                        shuffle=True)
    
    def predict(self):
        step_size_train = self.test_generator.n//self.test_generator.batch_size
        filenames = self.test_generator.filenames
        nb_samples = len(filenames)
        pred= self.model.predict_generator(self.test_generator,steps = nb_samples)
        test_loss, test_acc = self.model.evaluate_generator(
            generator=self.test_generator, steps=step_size_train)
        # print(test_loss, test_acc )
        return pred, test_loss, test_acc
        
        
    