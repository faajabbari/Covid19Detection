import os 
from csv import reader
import cv2
from shutil import copyfile
from funcy import flatten, isa
from distutils.dir_util import copy_tree
import argparse


def read_csv(csv_path):

    with open(csv_path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        #print(list_of_rows_train)
        print(len(list_of_rows))
    return list_of_rows

def copy_files(list_of_rows_train, dst_folder, subset, copy_type):

    for i in range(len(list_of_rows_train)):
        try:
            tt = list_of_rows_train[i][0][2:-2]
            label = tt.split('/')[6]
            dst = os.path.join(dst_folder, subset)#'/mnt/data/dataset_window_without_omiting_closed_lungs/train/'
            if copy_type == 'file':
                ss = tt.split('/')[5] + '__' + tt.split('/')[6] + '__' + tt.split('/')[7]
                if label == 'converted_without_windownon_covid':
                    copyfile(tt, os.path.join(dst, 'normal',  ss))
                else:
                    copyfile(tt, os.path.join(dst, 'covid', ss))
            else:
                ss = tt.split('/')[-2] + '__' + tt.split('/')[-1]
                #osmkdir(os.path.join(dst, 'normal', ss))
                if label == 'chest_pathology':
                    copy_tree(tt, os.path.join(dst, 'covid', ss))
                else:
                    copy_tree(tt, os.path.join(dst, 'normal', ss))
        except Exception as e:
            print(e)

def compute_intersection(list1, list2):

    a = flatten(list1)
    b = flatten(list2)
    common_elements = list(set(a).intersection(set(b)))
    return common_elements

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default='dataset_without_window_without_omiting_closed_lungs_100_sample_test')
    parser.add_argument("--output_folder", default='/mnt/data/100samples')
    args = parser.parse_args()
    
    folder = args.input_folder
    list_of_rows_test =  read_csv(os.path.join(folder, 'combine', 'test_set_final.csv'))
    
    dst_folder = args.output_folder
    copy_files(list_of_rows_test, dst_folder, 'test', 'folder')

