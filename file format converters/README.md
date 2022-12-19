**Install dicom2nifti:** `pip install dicom2nifti`

**Convert dcm files to nii.gz:** `dicom2nifti -G -r -o 1 -p -1000 input_directory output_directory`

**Install med2image:** `pip install med2image`

**Convert nii file to jpg format:**`med2image -i p2/3_lung_60__u91s.nii  -d input_healthy`


**How to read dicom files in python:**

Two python libraries have been deployed:
1. _pydicom_:


Installation: pip install pydicom

- Shifts the values so that the minimum intensity is 0
- Rotate image 90 degrees counterclockwise


2. _pymed_:


Installation: pip install pymed

- keeps values in Hounsfield scale

