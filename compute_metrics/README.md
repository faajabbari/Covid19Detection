**Describtion:**


- metric_check3: evaluates metrics over different thresholds from 0 to 1 by 0.01 steps. draws ROC curve over #covid/#totall and mean of confidances
- metrics_check4: evaluates metrics over 2 threshold by given intervals by 0.01 steps. draws ROC curve over #covid/#totall or mean of confidences
- compute_metrics: draws the distribution of probabilities


_input:_ csv s (output of inferences code)


_metrics:_ accuracy, recall, percision, f1_score


**results:**

1. by drawing ROC curve an d observing AUC of #covid/#totall and mean of confidences, #covid/#total is the better evaluator

2. distribution of of probabilities is non-gaussian, therefore the mean and standard deviation of 2 classes is not a appropriate for finding threshold. So we shifted the thresholds by 0.01
3. in two threshold method, optimum thresholds are: 0.13, 0.2 about 7% of datas are in suspicious area


**How to run:**

`python metric_check3.py --csv_dir --save_dir`

`python metric_check4.py --csv_dir --save_dir --th1 --th2`

`python compute_metrics.py --csv_dir`


