'''
input:
    main_dir
    model:
    th: 

output:
    metrics (case level): accuracy, f1_score, precision, recall
    Several csv files will be saved. (It cat be used in .py for finding
    optimum threshold):

    percentage_covid
    percentage_normal
    confidence_covid
    confidence_norml
    slice_level_prob_covid
    slice_level_prob_normal
    slice_level_names_covid
    slice_level_names_normal

'''

import csv
import os
import glob
import shutil
import math
import argparse

from keras.models import load_model
import tensorflow as tf
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, \
        accuracy_score, confusion_matrix

#from ct_selection_alg_def import ct_selection_alg


import pudb; pu.db
def read_csv(csv_path):
    with open(csv_path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        #print(list_of_rows_train)
        print(len(list_of_rows))
        return list_of_rows


def write_csv(list_csv, case):
    with open(case + '_final.csv', 'w') as f:
        writer = csv.writer(f)
        for i in list_csv:
            writer.writerow([i])


def prediction(cases, th , label, omit_closed_lung=True, smooth=True):
    #import pudb; pu.db
    covid_percentage_list = []
    confidence_list = []
    covid_slices_list = []
    labels_t = []
    preds_t = []
    p_softmax_list = []
    name_list = []
    for person_folder in cases:
        slides = sorted(glob.glob(person_folder+'/*'))
        names = slides
        preprocces_slides = []
        for slide in slides:
            img = cv2.imread(slide)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocces_slides.append(img)
        if omit_closed_lung == True:
            #preprocces_slides = omit_close_lung_window(preprocces_slides)
            start = math.ceil(0.2 * len(preprocces_slides))
            endd = len(preprocces_slides) - math.ceil(0.25 * len(preprocces_slides))
            preprocces_slides = preprocces_slides[start:endd]
            names = names[start:endd]
        else:
            preprocces_slides = preprocces_slides

        predictions = model.predict(np.array(preprocces_slides))
        import pudb; pu.db
        #import pudb;pu.db
        #predictions1 = tf.nn.softmax(predictions)
        
        if smooth != True:
            predictions = tf.nn.sigmoid(predictions)
            predictions = tf.where(predictions < 0.5, 0, 1)
            covid_slices = list(predictions).count(1)
            normal_slices = list(predictions).count(0)
        else:
            predictions = tf.nn.softmax(predictions)
            p = []
            p_softmax = []
            for element in predictions:
                p.append(np.argmax(element))
                p_softmax.append(element.numpy()[1])
            covid_slices = list(p).count(1)
            normal_slices = list(p).count(0)

        covid_percentage = covid_slices / len(predictions)
        print(covid_percentage)
        covid_confidence = sum(p_softmax)/len(p_softmax)
        print(covid_confidence)


        if covid_percentage > th:
            pred = 1  # 'covid'
        else: 
            pred = 0 # 'normal'

        labels_t.append(label)
        preds_t.append(pred)
        covid_percentage_list.append(covid_percentage)
        confidence_list.append(covid_confidence)
        covid_slices_list.append(covid_slices)
        p_softmax_list.append([p_softmax])
        name_list.append([slides])
    return covid_percentage_list, covid_slices_list, labels_t, preds_t, p_softmax_list, \
            name_list, confidence_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--main_dir', required=True,
            help='dir to test set containes cases each of which has \
            between 200-300 dicom files')
    parser.add_argument('--model', required=True,
                        help='pass to trained model')
    parser.add_argument('-t', '--threshold', required=True, type=float,
                        help='threshohd')
    parser.add_argument('--csv_pth', required=True, default='csvs/th_v3',
                        help='path to file to write csv s')
    
    args = parser.parse_args()
    
    
    
    main_dir = args.m
    # Ex: '/mnt/data/dataset_without_window_without_omiting_closed_lungs_20_30/th'
    model = load_model(args.model)
    # Ex:'/mnt/ssd/covid/transfere_learning _scripts/densenet_models/densenet_smooth_label
    #/my_model_densenet_fine_tune_smooth_label.h5')
    th  = args.t # 0.13

    folders = ['normal','covid']
    patients = sorted(os.listdir(os.path.join(main_dir, folders[1])))
    impatients = sorted(os.listdir(os.path.join(main_dir, folders[0],)))

    #import pudb;pu.db
    patients = list(map(lambda x: os.path.join(main_dir, folders[1], x), patients))
    impatients = list(map(lambda x: os.path.join(main_dir, folders[0], x), impatients))

    patients_covid_percentage, patients_covid_slices, l_c, p_c, p_softmax_covid, names_covid, \
            confidence_covid = prediction(patients, th=th, label=1) #'covid')
    impatients_covid_percentage, impatients_covid_slices, l_n, p_n, p_softmax_normal,names_normal, \
            confidence_normal =  prediction(impatients, th=th, label=0) #'normal')
    import pudb; pu.db
    print('covid-->', 'mean:', sum(patients_covid_percentage)/len(patients_covid_percentage), \
            'std: ', np.std(patients_covid_percentage) , \
            'min:',  min(patients_covid_percentage), \
            'max', max(patients_covid_percentage))
    print('normal-->', 'mean:', sum(impatients_covid_percentage)/len(impatients_covid_percentage), \
            'std: ', np.std(impatients_covid_percentage), \
            'min:', min(impatients_covid_percentage), \
            'max', max(impatients_covid_percentage))
    
    predss = p_c + p_n
    labelss = l_c + l_n
    #import pudb; pu.db
    acc = accuracy_score(labelss, predss)
    precision = precision_score(labelss, predss)
    recall = recall_score(labelss, predss)
    f1_score = f1_score(labelss, predss)
    confusion_matrix = confusion_matrix(labelss, predss)
    #import pudb; pu.db
    print('accuracy: '+ str(acc))
    print('f1_score: '+ str(f1_score))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print(confusion_matrix)
    write_csv(patients_covid_percentage, os.path.join(args.csv_pth,'percentage_covid'))
    write_csv(impatients_covid_percentage, os.path.join(args.csv_pth,'percentage_normal'))
    
    write_csv(confidence_covid, os.path.join(args.csv_pth,'confidence_covid'))
    write_csv(confidence_normal, os.path.join(args.csv_pth,'confidence_normal'))
    
    write_csv(p_softmax_covid, os.path.join(args.csv_pth,'slice_level_prob_covid'))
    write_csv(p_softmax_normal, os.path.join(args.csv_pth,'slice_level_prob_normal'))
    
    write_csv(names_covid, os.path.join(args.csv_pth,'slice_level_names_covid'))
    write_csv(names_normal, os.path.join(args.csv_pth,'slice_level_names_normal'))
