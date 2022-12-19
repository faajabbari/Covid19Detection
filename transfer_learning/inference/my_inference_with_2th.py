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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# from ct_selection_alg_def import ct_selection_alg


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

def omit_close_lung_window(series):
    all_path = []
    import pudb; pu.db
    paths = ct_selection_alg(series, 'jfjkh')
    return paths #all_path


def prediction(cases, th1, th2 , label, omit_closed_lung=True, smooth=True):
    import pudb; pu.db
    covid_percentage_list = []
    covid_slices_list = []
    labels_t = []
    preds_t = []
    for person_folder in cases:
        slides = sorted(glob.glob(person_folder+'/*'))
        preprocces_slides = []
        for slide in slides:
            img = cv2.imread(slide)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preprocces_slides.append(img)
        if omit_closed_lung == True:
            #preprocces_slides = omit_close_lung_window(preprocces_slides)
            start = math.ceil(0.2 * len(preprocces_slides))
            endd = len(preprocces_slides) - start
            preprocces_slides = preprocces_slides[start:endd]
        else:
            preprocces_slides = preprocces_slides

        predictions = model.predict(np.array(preprocces_slides))
        import pudb;pu.db
        #predictions1 = tf.nn.softmax(predictions)
        
        if smooth != True:
            predictions = tf.nn.sigmoid(predictions)
        else:
            predictions = tf.nn.softmax(predictions)
            import pudb;pu.db
        predictions = tf.where(predictions < 0.5, 0, 1)
        import pudb;pu.db
        p = []
        for element in predictions:
            p.append(np.argmax(element))
        covid_slices = list(p).count(1)
        normal_slices = list(p).count(0)
        covid_percentage = covid_slices / len(predictions)
        print(covid_percentage)
        if covid_percentage >= th2 :
            pred = 0  # 'covid'
            labels_t.append(label)
            preds_t.append(pred)
        elif covid_percentage <= th1: 
            pred = 1 # 'normal'
            labels_t.append(label)
            preds_t.append(pred)
        else:
            import pudb;pu.db
            print('mashkook')


        covid_percentage_list.append(covid_percentage)
        covid_slices_list.append(covid_slices)
    return covid_percentage_list, covid_slices_list, labels_t, preds_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--main_dir', required=True,
    help='dir to test set containes cases each of which has \
            between 200-300 dicom files')
    parser.add_argument('--model', required=True,
            help='pass to trained model')
    parser.add_argument('--th1', '--threshold1', required=True, type=float,
            help='first threshohd')
    parser.add_argument('--th2', '--threshold2', required=True, type=float,
            help='second threshold')
    parser.add_argument('--csv_pth', required=True, default='csvs/th_v3',
            help='path to file to write csv s')
    
    args = parser.parse_args()
    main_dir = args.main_dir
    # Ex: '/mnt/data/dataset_without_window_without_omiting_closed_lungs_20_30/th'
    model = load_model(args.model)
    # Ex:'/mnt/ssd/covid/transfere_learning _scripts/densenet_models/densenet_smooth_label
    #/my_model_densenet_fine_tune_smooth_label.h5')
    th1 = args.th1 # 0.13
    th2 = args.th2 # 0.2
    
    folders = ['normal','covid']
    patients = sorted(os.listdir(os.path.join(main_dir, folders[1])))
    impatients = sorted(os.listdir(os.path.join(main_dir, folders[0],)))
    patients = list(map(lambda x: os.path.join(main_dir, folders[1], x), patients))
    impatients = list(map(lambda x: os.path.join(main_dir, folders[0], x), impatients))

    patients_covid_percentage, patients_covid_slices, l_c, p_c = prediction(patients, th1=th1, th2=th2, label=0) #'covid')
    impatients_covid_percentage, impatients_covid_slices, l_n, p_n = prediction(impatients, th1=th1, th2=th2, label=1) #'normal')
    
    print('covid-->', 'mean:', sum(patients_covid_percentage)/len(patients_covid_percentage), 'min:',  min(patients_covid_percentage), 'max', max(patients_covid_percentage))
    print('normal-->', 'mean:', sum(impatients_covid_percentage)/len(impatients_covid_percentage), 'min:', min(impatients_covid_percentage), 'max', max(impatients_covid_percentage))
    
    predss = p_c + p_n
    labelss = l_c + l_n
    acc = accuracy_score(labelss, predss)
    precision = precision_score(labelss, predss)
    recall = recall_score(labelss, predss)
    f1_score = f1_score(labelss, predss)
    confusion_matrix = confusion_matrix(labelss, predss)
    print('accuracy: '+ str(acc))
    print('f1_score: '+ str(f1_score))
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print(confusion_matrix)
    write_csv(patients_covid_percentage, os.path.join(args.csv_pth,'covid_percentage'))
    write_csv(impatients_covid_percentage, os.path.join(args.csv_pth,'normal_percentage'))
    
    write_csv(patients_covid_slices, os.path.join(args.csv_pth, 'covid_slices'))
    write_csv(impatients_covid_slices, os.path.join(args.csv_pth,'normal_slices'))
