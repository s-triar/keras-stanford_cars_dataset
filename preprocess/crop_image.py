import cv2
import os
import pandas as pd


class CropImage():

    def __init__(self, path_source, path_destination, path_annotation_test, path_annotation_train):
        self.path_source = path_source
        self.path_destination = path_destination
        self.path_annotation_test = path_annotation_test
        self.path_annotation_train = path_annotation_train
        self.annotation = None
        self.check = ""
    
    def run(self):
        return self.crawl()

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
        if iteration == total:
            print()

    def openAnnotation(self, path):
        self.annotation = pd.read_csv(path, header=None)

    def crop(self, name, *args):
        try:
            if(args[0] != self.check):
                self.check = args[0]
                if(self.check == "train"):
                    self.openAnnotation(self.path_annotation_train)
                else:
                    self.openAnnotation(self.path_annotation_test)
        except:
            self.openAnnotation(self.path_annotation_test)

        img = cv2.imread(os.path.join(self.path_source, *args, name))
        data = self.annotation.loc[(self.annotation[0] == name)]
        hAwal = data[2].values[0]
        hAkhir = data[4].values[0]
        wAwal = data[1].values[0]
        wAkhir = data[3].values[0]
        crop_img = img[hAwal:hAkhir, wAwal:wAkhir]
        path = os.path.join(self.path_destination, *args)
        if(not os.path.exists(path)):
            os.makedirs(path)
        cv2.imwrite(os.path.join(self.path_destination, *args, name), crop_img)

    def crawl(self):
        if(os.path.exists(self.path_destination)):
            return "already cropped"
        else:
            os.makedirs(self.path_destination)
            for folder in os.listdir(self.path_source):
                if(os.path.isdir(os.path.join(self.path_source, folder))):
                    sfolder = os.listdir(
                        os.path.join(self.path_source, folder))
                    for sub_folder in sfolder:
                        if(os.path.isdir(os.path.join(self.path_source, folder, sub_folder))):
                            sfolder2 = os.listdir(os.path.join(
                                self.path_source, folder, sub_folder))
                            i = 0
                            self.printProgressBar(
                                i, len(sfolder2), prefix='Progress:', suffix='Complete', length=50)
                            for sub_folder2 in sfolder2:
                                if(os.path.isdir(os.path.join(self.path_source, folder, sub_folder, sub_folder2))):
                                    print("still directories in :",
                                          folder, sub_folder, sub_folder2)
                                else:
                                    text = "Progress: {} kelas: {} image: {}".format(
                                        folder, sub_folder, sub_folder2)
                                    self.printProgressBar(
                                        i + 1, len(sfolder2), prefix=text, suffix='Complete', length=50)
                                    i+=1
                                    self.crop(sub_folder2, folder, sub_folder)
                        else:
                            self.crop(sub_folder, folder)
                else:
                    self.crop(folder)
            return "crop process is finished"




# # path untuk dataset asli - dari folder parent test dan train
# source = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/car_data"
# # path untuk dataset hasil crop dengan annotation
# destination = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/cropped/car_data"
# # path annotation dataset test
# # jika test dan train jadi satu isi dengan annotation test
# anno_test = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_test.csv"
# # path annotation dataset train
# anno_train = "G:/dataset_mobil/stanford2-car-dataset-by-classes-folder/anno_train.csv"

# a = CropImage(source, destination, anno_test, anno_train)
# res = a.crawl()
# print(res)
