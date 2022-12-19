**Convert dicom files to png (window) dataset v3 Covid:**

`python dataset_convert_lable_server_window.py <csv file> <directory of all cases>`

**Example:**

`python dataset_convert_lable_server.py covid_40-_---.csv  /mnt/data/covid_our_data_v3/dataset\ covid`

(Notice: we used cases that were over 40% covid)

**Convert dicom files to png (window) dataset v3 Non Covid:**

`python dataset_convert_lable_server_non_covid.py <csv file> <directory of all cases>`

**Convert dicom files to png (without window) dataset v3 Covid:**

`python dataset_convert_lable_server.py <csv file> <directory of all cases>`

**Convert dicom files to png (without window) dataset v3 Non Covid:**

`python dataset_convert_lable_server_non_covid_without_window.py <csv file> <directory of all cases>`

**Example:**

`python dataset_convert_lable_server_non_covid_without_window.py  ./non_covid.csv /mnt/data/covid_our_data_v3/dataset\ covid`

**Convert dicom files to png dataset v2:**

`python data_converter_100_sample_test.py <directory of all cases>`
  
**Example:**

`python data_converter_100_sample_test.py /home/bahar/datasets/ct_scan/our_data_01_28_v2/chest_pathology/chest_covid_dicom`


---------------------------------------------------------------------------------------------------------------------------------


**Note:**
difference of between covid and non covid files:

for covid cases: `slice_label = 'covid' if file.rstrip('.dcm') in set(original_name) else 'non_covid'`

for normal cases: `slice_label = 'non_covid'`
