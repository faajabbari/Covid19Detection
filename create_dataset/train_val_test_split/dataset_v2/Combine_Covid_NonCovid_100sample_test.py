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
       import pudb; pu.db
    with open(os.path.join(folder , 'combine', subset + '___set_final.csv'), 'w') as f:
        writer = csv.writer(f)
        for i in random_normal:
            writer.writerow([i])



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default='dataset_without_window_without_omiting_closed_lungs_100_sample_test')
    args = parser.parse_args()
                            
    folder = args.input_folder

    test_covid = read__csv(folder + '/covid/test_set_covid.csv')
    test_normal = read__csv(folder + '/normal/test_set_normal.csv')

    test_set_random_covid = random_sample(test_normal, test_covid)
    combine_csv(test_normal, test_set_random_covid, 'test')
