**Description:**
1. my_inference_with_th_original: takes main_dir which contains of covid and normal sub folder. Each of subfolders has different samples contains  dicom files and computes metris over normal and covid samples with 1 threshold.
2. my_inference_with_2th: takes main_dir which contains covid and normal sub folder. Each of subfolders has different samples containe of dicom files and computes metris over normal and covid samples with 2 thresholds.

outputs:
- metrics (case level): accuracy, f1_score, precision, recall

- Several csv files will be saved. (It cat be used in .py for finding optimum threshold):

- percentage_covid
- percentage_normal 
- confidence_covid
- confidence_norml
- slice_level_prob_covid
- slice_level_prob_normal
- slice_level_names_covid
- slice_level_names_normal



3. my_inference_with_th_for_one_case_dicom_edited: take a directory which contains all the samples belong to a person and predicts whether the case is covid positive or not.




