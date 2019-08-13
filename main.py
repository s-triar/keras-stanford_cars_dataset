from preprocess.crop_image import CropImage
from training import DeepLearning_training
from testing import DeepLearning_testing


def main_learning(runningN, path_train, path_test, epochs, batch_size, model):
    learning = DeepLearning_training(
        runningN, path_train, path_test, epochs, batch_size, model)
    learning.run_training()


def main_testing(path_model, path_test, batch_size):
    testing = DeepLearning_testing(path_model, path_test, batch_size)
    prediction, test_loss, test_acc = testing.predict()
    print("prediction", prediction)
    print("test loss", test_loss)
    print("test accuracy", test_acc)

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
    path_train = './data/train'
    # path untuk data testing
    path_test = './data/test'
    # jumlah epoch
    epochs = 200
    # ukuran batch
    batch_size = 32
    # nama model yang digunakan. Ada: mobilenet, mobilenetv2, vgg19, resnet50
    model = "mobilenetv2"
    # nama running learning. Digunakan untuk nama pada hasil model dan visualisasi di tensorboard 
    run_name = model
    # path model yang sudah ditraining
    path_model = './models/mobilenetv2_car_indo1565703295.8897007.h5'

    # melakukan proses training jika true, jika tidak melakukan proses testing
    train = True
    if(train):
        # crop hanya jika memakai data stanford cars dataset
        # prep = CropImage(source, destination, anno_test, anno_train)
        # prep.run()
        main_learning(run_name, path_train, path_test,
                      epochs, batch_size, model)
    else:
        main_testing(path_model, path_test, batch_size)
