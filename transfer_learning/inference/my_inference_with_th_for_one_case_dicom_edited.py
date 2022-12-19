import csv
import os
import glob
import math
import argparse

import png
import pydicom
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np


def duplicates(lst, item):

    return [i for i, x in enumerate(lst) if x == item]


def dicom_to_png(filenames):
    '''
    input: list of dicom filenames.
    output: a dictionary that consists of : key-->seri number, value-->list of numpy array with same seri number.
    '''
    seri_numbers = []
    outputs = []
    series = []
    for filename in filenames:
        try:
            import pudb; pu.db
            ds = pydicom.dcmread(filename)
            image_2d = ds.pixel_array.astype(float) 
            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)
            default = str(ds.WindowCenter[0])
            ff = 's' + str(ds.SeriesNumber) + '___' + str(ds.InstanceNumber).zfill(3) + '___' + 'w' + default + '___' + str(ds.PatientID)
            f1, f2 = os.path.split(filename)
            if ds.SeriesNumber not in series:
                series.append(ds.SeriesNumber)
                os.makedirs(os.path.join(f1, 'inf', str(ds.SeriesNumber)), exist_ok=True)

            out = os.path.join(f1, 'inf', str(ds.SeriesNumber), ff + f2.rstrip('.dcm')+ '.png')

            with open(out, 'wb') as png_file:
                w = png.Writer(512, 512, greyscale=True)
                w.write(png_file, image_2d_scaled)
                
            img = cv2.imread(out)
            image_2d_scaled = img
            sn = ds.SeriesNumber
            seri_numbers.append(sn)
            outputs.append(image_2d_scaled)
        except Exception as e:
            print(e)
    seri_unique = np.unique(np.array(seri_numbers))
    output = {}
    for s in seri_unique:
        
        sn_indexes = duplicates(seri_numbers, s)
        sn_numpy = []
        for sn in sn_indexes:
            sn_numpy.append(outputs[sn])
        output[sn] = sn_numpy
    
    return output


def pick_larger_sery(folder):
    length = []
    largest_sery = 0
    for key, value in folder.items():
        ser = len(value)
        if ser > largest_sery:
            largest_sery = ser
            key_largest_sery = key
    
    return folder[key_largest_sery]


def prediction(case, th1, th2 , omit_closed_lung=True, smooth=True):
    
    slides = sorted(glob.glob(case+'/*.dcm'))
    preprocces_slides1 = dicom_to_png(slides)
    preprocces_slides = pick_larger_sery(preprocces_slides1)

    if omit_closed_lung == True:

        start = math.ceil(0.2 * len(preprocces_slides))
        endd = len(preprocces_slides) - math.ceil(0.25 * len(preprocces_slides))
        import pudb; pu.db
        preprocces_slides = preprocces_slides[start:endd]

    else:

        preprocces_slides = preprocces_slides

    predictions = []
    for i in range(len(preprocces_slides)):
        pred = model.predict(np.array(np.expand_dims(preprocces_slides[i], axis=0)))
        predictions.append(pred)
        
    if smooth != True:

        predictions = tf.nn.sigmoid(predictions)
        covid_slices = list(predictions).count(0)
        normal_slices = list(predictions).count(1)
        covid_slices_index = duplicates(list(predictions), 0)
 
    else:

        predictions = tf.nn.softmax(predictions)

        p = []
        p_covid = []
        for element in predictions: 
            p_covid.append(element[0][1])
            p.append(np.argmax(element))
       
        predictions = p
        covid_slices = list(p).count(1)
        normal_slices = list(p).count(0)
        covid_slices_index = duplicates(list(predictions), 1)
    
    covid_percentage = covid_slices / len(predictions)

    if covid_percentage >= th2: 
        pred = 'covid'
    
    elif covid_percentage <= th1:
        pred = 'normal'

    else:
        pred = 'Suspicious'
    

    covid_slices_address = []
    covid_covid_probe = []
    normal_slices_address = []
    normal_covid_probe = []
    for i in range(len(p_covid)):
        if i in covid_slices_index:
            covid_slices_address.append(slides[i])
            covid_covid_probe.append(p_covid[i])
        else:
            normal_slices_address.append(slides[i])
            normal_covid_probe.append(p_covid[i])

    return pred, covid_slices_address, normal_slices_address, covid_covid_probe,\
            normal_covid_probe, covid_percentage

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', required=True,\
            help='pass to main directory')
    parser.add_argument('--model_dir', required=True,\
            help='pass to trained model directory')
    parser.add_argument('--th1', required=True,\
            help='first threshold')
    parser.add_argument('--th2', required=True,\
            help='second threshold')
    args = parser.parse_args()

    main_dir = args.maina_dor 
    # Ex:'/home/bahar/datasets/ct_scan/our_data_01_28_v2/chest_pathology/chest_covid_dicom/16'
    model = load_model(args.model_dir)
    # Ex:('/mnt/ssd/covid/transfere_learning _scripts/densenet_models/densenet_without_window_20_30\
    #        /my_model_densenet_fine_tune_smooth_labelwithout_window_20_30.h5')
    th1 = args.th1
    th2 = args.th2

    pred, covid_slices_address, normal_slices_address, covid_covid_probe, normal_covid_probe, covid_percentage =\
            prediction(main_dir, th1, th2) # 0.21, 0.32)
    print(pred, len(covid_slices_address), covid_percentage)

