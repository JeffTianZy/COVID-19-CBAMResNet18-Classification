import os
import pandas as pd
import cv2 as cv
def load_data(path):

    data_csv = pd.read_csv(os.path.join(path, 'Chest_xray_Corona_Metadata.csv'))

    #os.mkdir(path + './Train')
    os.mkdir(path + './Test')

    image_name = data_csv['X_ray_image_name']
    dataset_type = data_csv['Dataset_type']
    labels = data_csv['Label']
    is_covid = data_csv['Label_2_Virus_category']

    for i in range(len(image_name)):
        if dataset_type[i] == 'TRAIN':
            """img = cv.imread(os.path.join(path, 'Coronahack-Chest-XRay-Dataset/train', image_name[i]))
            resize_img = cv.resize(img, (224, 224))
            if labels[i] == 'Normal':
                rename = 'Normal_' + image_name[i]
                newpath = os.path.join(path, 'Train', rename)
                cv.imwrite(newpath, resize_img)
            if is_covid[i] == 'COVID-19':
                rename = 'Covid_' + image_name[i]
                newpath = os.path.join(path, 'Train', rename)
                cv.imwrite(newpath, resize_img)"""
            pass
        if dataset_type[i] == 'TEST':
            img = cv.imread(os.path.join(path, 'Coronahack-Chest-XRay-Dataset/test', image_name[i]))
            resize_img = cv.resize(img, (224, 224))
            if labels[i] == 'Normal':
                rename = 'Normal_' + image_name[i]
                newpath = os.path.join(path, 'Test', rename)
                cv.imwrite(newpath, resize_img)
            if is_covid[i] == 'COVID-19':
                rename = 'Covid_' + image_name[i]
                newpath = os.path.join(path, 'Test', rename)
                cv.imwrite(newpath, resize_img)
path = "./Data/X-ray/coronahack-chest-xraydataset"
load_data(path)