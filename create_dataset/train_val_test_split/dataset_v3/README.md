**Build dataset**

**csv output samples:  [google drive](https://drive.google.com/file/d/183-bfFpygVOMRXhNRXdePsxQbf9IsdW_/view?usp=sharing)**

**Step 1: for splitting samples for train, validation, threshold and test:**

`mkdir <csv files>`

`mkdir <csv files>/covid`

`mkdir <csv files>/normal`

`mkdir <csv files>/combine`

(Notice: `<csv files>` is a name of directory that consists of outputs csv)

`python train_val_test_split.py --input_folder <input folder of png files> --percentage '[0.45, 0.1, 0.15, 0.3]' --case covid --output_folder <csv files> --omit_lung True` 

`python train_val_test_split.py --input_folder <input folder of png files> --percentage '[0.45, 0.1, 0.15, 0.3]' --case normal --output_folder <csv files> --omit_lung True`

**Notes:**

--case: Normal or covid

--percentage: [train, val, th, test]

--omit_lung: True or False 

you can change omit closed lungs algorithm in 2 ways
1. omit percentages of first and end slices
    - **Problem:** not exact for some cases.
2. window algoritm (ct_selection_alg_def.py)
    - best parameter for window png 
    - best parameter for without window png
    - **Problem:** data loss!

**Step 2: combine covid and normal csv and save final csv in `<csv files>/combine`**


**Example:**
`python Combine_Covid_NonCovid.py --input_folder <input directory (output csv of step 1)> --output_folder <output directory>`


**Step 3: check and copy files from final files**

`python check.py --input_folder <input directory (output csv of step 2)> --output_folder <output directory>`
