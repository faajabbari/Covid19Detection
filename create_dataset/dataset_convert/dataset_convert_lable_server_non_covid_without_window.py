import os

import numpy as np
import png
import shutil
import glob
# import cv2
import pydicom
import sys
from medpy.io import load
from tqdm import tqdm
import csv
import pudb; pu.db


def dicom2png(source_folder, file, series, output_folder):
    try:

        #image_2d, image_header = load(os.path.join(source_folder, file))
        ds = pydicom.dcmread(os.path.join(source_folder, file))
        image_2d = ds.pixel_array.astype(float)

        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
        # Convert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        # Convert to uint
        #image_2d_scaled = np.uint8(image_2d_scaled)
        image_2d_scaled = np.squeeze(image_2d_scaled)
        #image_2d_scaled = np.transpose(image_2d_scaled)
        file2w = image_2d_scaled.copy()
        default = str(ds.WindowCenter[0])
        case_label = 'non_covid' # source_folder.split('/')[-2]
        # slice_label = 'covid' if file.rstrip('.dcm') in set(original_name) else 'non_covid'
        slice_label = 'non_covid'
        ff = 's' + str(ds.SeriesNumber) + '___' + str(ds.InstanceNumber).zfill(3) + '___' + 'w' + default + '___' + str(ds.PatientID) + '__' + case_label + '__' + slice_label  # file.rstrip('.dcm')

        if ds.SeriesNumber not in series:
            series.append(ds.SeriesNumber)
            os.makedirs(os.path.join(output_folder, str(ds.SeriesNumber)), exist_ok=True)

        out = os.path.join(output_folder, str(ds.SeriesNumber), ff + '.png')


        with open(out, 'wb') as png_file:
            w = png.Writer(512, 512, greyscale=True)
            w.write(png_file, file2w)

    except:
        print('Could not convert: ', os.path.join(source_folder, file))



def dicom2png_case(source_folder, output_folder, album_id):
    global count_on_list
    global count_not_on_list
    global total_count
    total_count += 1
    if os.path.split(source_folder)[-1] in set(album_id):
        count_on_list +=1
        list_of_files = sorted(os.listdir(source_folder))
        count = 0
        series = []

        for file in list_of_files:
            dicom2png(source_folder, file, series, output_folder)

        created_folders = os.listdir(output_folder)
        for created_folder in created_folders:
            path_l = os.path.join(output_folder, created_folder)
            len_ct = len(os.listdir(path_l))
            list_ct = sorted(os.listdir(path_l))
            if len_ct < 2:
                shutil.rmtree(path_l)
    else:
        count_not_on_list += 1
    print(f'total:{total_count} , non_list : {count_not_on_list}, on_list: {count_on_list}')
        # a = len(series)


if __name__ == '__main__':

    import pudb; pudb
    album_id =[]
    # global count_on_list
    # global count_not_on_list
    count_on_list = 0
    count_not_on_list = 0
    total_count = 0
    # global count_on_list
    # global count_not_on_list
    # original_name = []
    csv_dir = sys.argv[1]
    with open(csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                album_id.append(row[0])
                # original_name.append(row[1].rstrip('.png'))
        print(f'Processed {line_count} lines.')
    import pudb; pudb
    main_dir = sys.argv[2]
    temp = os.path.split(main_dir)
    last_dir = os.path.join(temp[0], 'converted_without_window' + 'non_covid')
    os.makedirs(last_dir, exist_ok=True)
    import pudb; pudb

    cases = sorted(os.listdir(main_dir))
    cases = list(map(lambda x: os.path.join(main_dir, x), cases))
    for person_folder in tqdm(cases):
    # print(person)

        add = os.path.split(person_folder)[-1]
        save_path = os.path.join(last_dir, add)
#        os.makedirs(save_path, exist_ok=True)
        
        dicom2png_case(person_folder, save_path, album_id)
