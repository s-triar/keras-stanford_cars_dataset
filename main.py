from preprocess.crop_image import CropImage
from training import DeepLearning_training
from testing import DeepLearning_testing


def main_learning(runningN, path_train, path_test, epochs, batch_size, model):
    learning = DeepLearning_training(
        runningN, path_train, path_test, epochs, batch_size, model)
    learning.run_training()


def main_testing(path_model, path_test, batch_size):
    testing = DeepLearning_testing(path_model, path_test, batch_size)
    prediction = testing.predict()
    print(prediction)


if __name__ == "__main__":
    # path untuk dataset asli - dari folder parent test dan train
    source = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/car_data"
    # path untuk dataset hasil crop dengan annotation
    destination = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data"
    # path annotation dataset test
    # jika test dan train jadi satu isi dengan annotation test
    anno_test = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_test.csv"
    # path annotation dataset train
    anno_train = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_train.csv"

    # path untuk data training
    path_train = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data/train'
    # path untuk data testing
    path_test = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/car_data/test'
    # jumlah epoch
    epochs = 50
    # ukuran batch
    batch_size = 32
    # nama model yang digunakan. Ada: mobilenet, mobilenetv2, vgg19, resnet50
    model = "mobilenetv2"
    # nama running learning. Digunakan untuk nama pada hasil model dan visualisasi di tensorboard 
    run_name = model
    # path model yang sudah ditraining
    path_model = 'G:/DTSTask/project1_classifier/models/MobileNetV21565453004.3961172.h5'

    # melakukan proses training jika true, jika tidak melakukan proses testing
    train = False
    if(train):
        prep = CropImage(source, destination, anno_test, anno_train)
        prep.run()
        main_learning(run_name, path_train, path_test,
                      epochs, batch_size, model)
    else:
        main_testing(path_model, path_test, batch_size)
