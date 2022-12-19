import os
import csv

import numpy as np
import matplotlib.pyplot as plt


def read_csv(csv_dir, csv_file):
    with open(os.path.join(csv_dir, csv_path), 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        return list_of_rows

def sum_of_list(l):
    total = 0
    for val in l:
        total = total + float(val[0])
    return total

def make_list(l):
    s = []
    for i in range(len(l)):
        s.append(float(l[i][0]))
    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_dir', required=True, help='addres to csv directories')
    args = parser.parse_args()


    csv_dir = args.csv_dir 
    covid_percentage_final = make_list(read_csv(csv_dir, 'covid_percentage_final.csv'))
    print('covid_percentage_final:',\
            'min:', min(covid_percentage_final),\
            'max:', max(covid_percentage_final),\
            'mean:', sum(covid_percentage_final)/len(covid_percentage_final ),#sum_of_list(covid_percentage_final),\
            'std:', np.std(np.array(covid_percentage_final)))
    
    normal_percentage_final = make_list(read_csv(csv_dir, 'normal_percentage_final.csv'))
    print('normal_percentage_final:',\
            'min:', min(normal_percentage_final),\
            'max:', max(normal_percentage_final),\
            'mean:', sum(normal_percentage_final)/len(normal_percentage_final),\
            'std:', np.std(np.array(normal_percentage_final)))
    
    
    covid_slices_final = make_list(read_csv(csv_dir,'covid_slices_final.csv'))
    print('covid_slices_final:',\
            'min:', min(covid_slices_final),\
            'max:', max(covid_slices_final),\
            'mean:', sum(covid_slices_final)/len(covid_slices_final),\
            'std:', np.std(np.array(covid_slices_final)))
    
    
    normal_slices_final = make_list(read_csv(csv_dir,'normal_slices_final.csv'))
    print('normal_slices_final:',\
            'min:', min(normal_slices_final),\
            'max:', max(normal_slices_final),\
            'mean:', sum(normal_slices_final)/len(normal_slices_final),\
            'std:', np.std(np.array(normal_slices_final)))
    
    plt.hist(normal_percentage_final, bins =5);plt.savefig('hist_normal.png') 
    plt.hist(covid_percentage_final, bins =5);plt.savefig('hist_covid.png')
