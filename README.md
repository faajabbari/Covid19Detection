# COVID-19 

## detection COVID-19 from chest CT-Scan


<!--- # Tasks: (for consideration)

1. whether augumentation is reasonable? (flip, rotation, ... )















_____________________________________________________________________________________________________________________
### Tasks:  ref:kaggle

1. COVID-19 Classification

Purpose:
Find evidence that the current image could present evidence of COVID-19 infection.
Note that the radiological findings of COVID are not exclusive to this disease.
Sugested Metrics:
    • AUROC
    • F1-Score

2. Lung Segmentation

Purpose:
Lung segmentation is often the first step to increase the performance of any other supervised model, as it takes away the noise created by regions of no interest for COVID-19.

Sugested Metric:
    • Dice Coefficient
    • IoU (aka Jaccard Index)

3.COVID-19 Infection Segmentation

Purpose
Segmentation of radiological findings can shed light into how severe the situation of the patient is, allowing for appropriate care.
Sugested Metrics
    • Dice Coefficient
    • IoU (aka Jaccard Index)

### The major concerns:   'ref: https://github.com/UCSD-AI4H/COVID-CT'

1. the quality of these images is degraded: The quality degradation includes: the Hounsfield unit (HU) values are lost; the number of bits per pixel is reduced; the resolution of images is reduced. 

2. the original CT scan contains a sequence of CT slices, but when put into papers, only a few key slices are selected, which may have negative impact on diagnosis as well. --->



## Directory structure

    ├── codes [checkout papers repo]
    ├── create_dataset
    ├── transfer_learning
    ├── compute_metrics
    ├── file format converters
    ├── datasets info
    ├── persian_doc
    └── README.md


---------------------------------------------------------------------------------------------------------------------------------

for training model and validate model on custom dicom dataset; follow the instructions

* **Step 1**

create dataset

    ├── dataset_convert
    ├── train_val_test_split
    ├── edge_detector
    ├── lung_extractor
    └── README.md

* **Step 2**

transfer_learning

    ├── train_val
    ├── inference
    └── README.md

* **Step 3**

compute_metrics

    ├── samples
    └── README.md
 


---------------------------------------------------------------------------------------------------------------------------------




