import os 
from csv import reader
import csv
import random

import argparse


def read__csv(csv_path):

    with open(csv_path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows_train = list(csv_reader)
        print(len(list_of_rows_train))
    return list_of_rows_train

def random_sample(covid_list, normal_list):
    n = len(covid_list)
    random_set = random.sample(normal_list, n)
    print(len(random_set))
    return random_set

def combine_csv(random_normal, train_covid, subset):
    for i in train_covid:
       random_normal.append(i)
    with open(os.path.join(folder , 'combine', subset + '_set_final.csv'), 'w') as f:
        writer = csv.writer(f)
        for i in random_normal:
            writer.writerow([i])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default='dataset_without_window_without_omiting_closed_lungs_20_30')
    parser.add_argument("--output_folder", default='dataset_without_window_without_omiting_closed_lungs_20_30')
    args = parser.parse_args()
                                
    folder = args.input_folder
    train_covid = read__csv(folder + '/covid/train_set_covid.csv')
    test_covid = read__csv(folder + '/covid/test_set_covid.csv')
    val_covid = read__csv(folder + '/covid/val_set_covid.csv')
    th_covid = read__csv(folder + '/covid/th_set_covid.csv')


    train_normal = read__csv(folder + '/normal/train_set_normal.csv')
    test_normal = read__csv(folder + '/normal/test_set_normal.csv')
    val_normal = read__csv(folder + '/normal/val_set_normal.csv')
    th_normal = read__csv(folder +'/normal/th_set_normal.csv')


    train_set_random_normal = random_sample(train_covid, train_normal)
    test_set_random_normal = random_sample(test_covid, test_normal)
    val_set_random_normal = random_sample(val_covid, val_normal)
    th_set_random_normal = random_sample(th_covid, th_normal)

    combine_csv(train_set_random_normal, train_covid, 'train')
    combine_csv(test_set_random_normal, test_covid, 'test')
    combine_csv(val_set_random_normal, val_covid, 'val')
    combine_csv(th_set_random_normal, th_covid,'th') 
