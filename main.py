from preprocess.crop_image import CropImage
from training import DeepLearning_training


def main(runningN, path_train, path_test, epochs, batch_size, model):
    learning = DeepLearning_training(runningN, path_train, path_test, epochs, batch_size, model)
    learning.run_training()

if __name__ == "__main__":
    # # path untuk dataset asli - dari folder parent test dan train
    source = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/car_data"
    # # path untuk dataset hasil crop dengan annotation
    destination = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data"
    # # path annotation dataset test
    # # jika test dan train jadi satu isi dengan annotation test
    anno_test = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_test.csv"
    # # path annotation dataset train
    anno_train = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_train.csv"

    prep= CropImage(source, destination, anno_test, anno_train)
    prep_res = prep.run()
    print(prep_res)

    
    path_train = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data/train'
    path_test = 'G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/car_data/test'
    epochs = 50
    batch_size = 32
    model="mobilenetv2"
    run_name=model

    main(run_name, path_train, path_test, epochs, batch_size, model)