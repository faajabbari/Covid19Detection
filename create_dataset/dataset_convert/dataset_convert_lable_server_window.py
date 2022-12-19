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
#import pudb; pu.db


def dicom2png(source_folder, file, series, output_folder):

    #import pudb; pu.db
   # try:
    if file == '1.3.12.2.1107.5.1.4.78983.30000020031604263609300162352.dcm':
        import pudb; pu.db
    image_2d, image_header = load(os.path.join(source_folder, file))
    ds = pydicom.dcmread(os.path.join(source_folder, file))

        # # lungs
    level = -600
    window = 1200
    max = level + window / 2
    min = level - window / 2
    image_2d[image_2d < min] = min
    image_2d[image_2d > max] = max
    image_2d = np.squeeze(image_2d)
    image_2d = image_2d.astype("float32")

    maxx = image_2d.max()
    minn = image_2d.min()
    image_2d_scaled = ((image_2d - minn) / (maxx - minn)) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    image_2d_scaled = np.squeeze(image_2d_scaled)
    image_2d_scaled = np.transpose(image_2d_scaled)
    file2w = image_2d_scaled.copy()
    default = str(ds.WindowCenter[0])
    case_label = source_folder.split('/')[-2]
    #import pudb; pu.db
    
    slice_label = 'covid' if file.rstrip('.dcm') in set(original_name) else 'non_covid'
    # slice_label = 'non_covid'
    ff = 's' + str(ds.SeriesNumber) + '___' + str(ds.InstanceNumber).zfill(3) + '___' + 'w' + default + '___' + str(ds.PatientID) + '__' + case_label + '__' + slice_label  # file.rstrip('.dcm')

    if ds.SeriesNumber not in series:
        series.append(ds.SeriesNumber)
        os.makedirs(os.path.join(output_folder, str(ds.SeriesNumber)), exist_ok=True)

    out = os.path.join(output_folder, str(ds.SeriesNumber), ff + '.png')


    with open(out, 'wb') as png_file:
        w = png.Writer(512, 512, greyscale=True)
        w.write(png_file, file2w)

    #except:
     #   print('Could not convert: ', os.path.join(source_folder, file))



def dicom2png_case(source_folder, output_folder, document_id, original_name):
    if os.path.split(source_folder)[-1] == '1.3.12.2.1107.5.1.4.78983.30000020031604245790600001430':
        import pudb; pu.db
    if os.path.split(source_folder)[-1] in set(document_id):
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


        a = len(series)


if __name__ == '__main__':

  
    document_id =[]
    original_name = []
    csv_dir = sys.argv[1]
    with open(csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for line_count, row in enumerate(csv_reader):
            # if line_count == 0:
            if False:
                print(f'Column names are {", ".join(row)}')
            else:
                document_id.append(row[0])
                original_name.append(row[1].rstrip('.png'))
        print(f'Processed {line_count} lines.')
    import pudb; pudb
    main_dir = sys.argv[2]
    temp = os.path.split(main_dir)
    last_dir = os.path.join(temp[0], 'converted_new' + temp[1])
    os.makedirs(last_dir, exist_ok=True)
    

    cases = sorted(os.listdir(main_dir))
    cases = list(map(lambda x: os.path.join(main_dir, x), cases))
   # cases = ['/mnt/data/covid_our_data_v3/dataset covid/1.3.12.2.1107.5.1.4.78983.30000020031604245790600001430']
    for person_folder in tqdm(cases):
        print(person_folder)

        add = os.path.split(person_folder)[-1]
        save_path = os.path.join(last_dir, add)
        print(save_path)
        #os.makedirs(save_path, exist_ok=True)

        dicom2png_case(person_folder, save_path, document_id, original_name)
