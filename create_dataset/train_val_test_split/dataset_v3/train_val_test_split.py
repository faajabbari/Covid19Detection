import os 
import sys
import math
import random
import glob
import csv
import argparse
from collections import OrderedDict
from ct_selection_alg_def import ct_selection_alg


def train_val_test_samples(main_dir, percentage):
    files = os.listdir(main_dir)
    total_len = len(files)
    train_len = int(math.ceil(percentage[0] * total_len))
    val_len = int(math.ceil(percentage[1] * total_len))
    threshold_len = int(math.ceil(percentage[2] * total_len))
    random.seed(777)
    train_samples = random.sample(files, train_len)
    val_th_test = list(set(files) - set(train_samples))
    val_samples = random.sample(val_th_test, val_len)
    th_test = list(set(val_th_test) - set(val_samples))
    th_samples = random.sample(th_test, threshold_len)
    test_samples = list(set(th_test) - set(th_samples))
    return train_samples, val_samples, th_samples, test_samples


def pick_larger_sery(main_dir, folder):
    res = []
    for f in folder:
        try:
            length = []
            p = os.path.join(main_dir, f)
            ser = os.listdir(p)
            for x in sorted(ser):
                length.append(len(os.listdir(os.path.join(p, x)))) # for x in os.path.join(main_dir, folder)]
        
            largest_sery = length.index(max(length))
            res.append(os.path.join(main_dir, f, sorted(ser)[largest_sery]))
        except Exception as e:
            print(e)
    return res


def omit_close_lung(series):
    final_res = []
    for sery in series:
        cts = sorted(os.listdir(sery))
        l = len(cts)
        l_omit_neck = math.ceil(0.15 * l)
        l_omit_stomack = math.ceil(0.26 * l)
        res = cts[l_omit_neck: l - l_omit_stomack]
        res2 = list(map(lambda x: os.path.join(sery, x), res))
        final_res = final_res + res2
    return final_res

def omit_close_lung_window(series):
    import time
    all_path = []
    for seri in series:
        t = time.time()
        paths = ct_selection_alg(seri, 'jfjkh')
        for path in paths:
            all_path.append(path)
    return all_path
    
def check_label(imgs):
    sample = []
    for img in imgs:
        img_path = os.path.split(img)[-1]
        labels = img_path.split('___')
        label = labels[-1]
        ll = label.split('__')[-1].split('.')[0]
        if ll == 'covid':
            sample.append(img)
    return sample


def get_files(folder_list):
    dicom_files = []
    for folder in folder_list:
        for decom_file in glob.glob(folder + '/*'):
            dicom_files.append(decom_file)
    return dicom_files

def get_files1(folder_list):
    dicom_files = []
    for folder in folder_list:
        for decom_file in glob.glob('/mnt/data/covid_our_data_v3/dataset covid/' + folder + '/*'):
            dicom_files.append(decom_file)
    return dicom_files

def write_csv(sample_list, folder, case, subset):
    if subset == 'train' or subset == 'val':
        with open(os.path.join(folder, case, subset + '_set_' + case + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for i in sample_list:
                writer.writerow([i])
    else:
        folders = []
        for i in sample_list:
            mm = i.split('/')
            folder_path = os.path.join('/' + mm[0], mm[1], mm[2], mm[3], mm[4], mm[5], mm[6])
            folders.append(folder_path)
        my_final_list = OrderedDict.fromkeys(folders)
        with open(os.path.join(folder, case, subset + '_set_' + case + '.csv'), 'w') as f:
            writer = csv.writer(f)
            for i in my_final_list:
                writer.writerow([i])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default='/mnt/data/covid_our_data_v3/converted_new_dataset_window_covid')
    parser.add_argument("--percentage", default=[0.45, 0.1, 0.15, 0.3])
    parser.add_argument("--case", default='covid')
    parser.add_argument("--output_folder", default='dataset_without_window_without_omiting_closed_lungs_20_30')
    parser.add_argument("--omit_lung", default=False)
    args = parser.parse_args()
    
    main_dir = args.input_folder
    folder = args.output_folder
    case = args.case
    percentage = args.percentage
    omit_lung = args.omit_lung
   

    train_samples, val_samples, th_samples, test_samples =  train_val_test_samples(main_dir, percentage)
    
    train_sample_l_sery = pick_larger_sery(main_dir, train_samples)
    val_sample_l_sery = pick_larger_sery(main_dir, val_samples)
    th_sample_l_sery = pick_larger_sery(main_dir, th_samples)
    test_sample_l_sery = pick_larger_sery(main_dir, test_samples)
    if omit_lung == True:
        train_set = omit_close_lung(train_sample_l_sery)
        val_set = omit_close_lung(val_sample_l_sery)
        th_set = omit_close_lung(th_sample_l_seryi)
        test_set = omit_close_lung(test_sample_l_sery)
    else:
        train_set = omit_close_lung(train_sample_l_sery)
        val_set = omit_close_lung(val_sample_l_sery)
        test_set = get_files(test_sample_l_sery)
        th_set = get_files(th_sample_l_sery)
    
    if case == 'covid':
        train_set = check_label(train_set)
        val_set = check_label(val_set)
        th_set = check_label(th_set)
        test_set = check_label(test_set)
    
    write_csv(train_set, folder, case, 'train')
    write_csv(test_set, folder, case, 'test')
    write_csv(val_set, folder, case, 'val')
    write_csv(th_set, folder, case, 'th')
