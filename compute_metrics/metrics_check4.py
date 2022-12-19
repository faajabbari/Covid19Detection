import os
import csv
import glob

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def read_csv(csv_path):
    with open(os.path.join(csv_path), 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
        #print(list_of_rows_train)
        #print(len(list_of_rows))
        return list_of_rows


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--csv_dir', required=True, help='addres to csv directories')
    parser.add_argument('-s', '--save_dir', required=True, help='addres to save resulted images')
    parser.add_argument('--th1', required=True, help='begginig of the interval')
    parser.add_argument('--th2', required=True, help='begginig of the interval')
    args = parser.parse_args()


    csv_dir = args.csv_dir  # Ex: 'csvs/th_v3'
    save_path = args.save_dir  # Ex: 'fig/v3_th'
    th1 = args.th1 # Ex: 0.13
    th2 = args.th2 # Ex: 0.2
    os.makedirs(save_path, exist_ok=True)
    
    # Reading csv files
    confidence_covid = np.array(read_csv(os.path.join(csv_dir, 'confidence_covid_final.csv')), dtype='float')
    confidence_normal = np.array(read_csv(os.path.join(csv_dir,'confidence_normal_final.csv')), dtype='float')
    confidence = np.concatenate((confidence_covid, confidence_normal), axis=0)
    
    percentage_covid = np.array(read_csv(os.path.join(csv_dir, 'percentage_covid_final.csv')), dtype='float')
    percentage_normal = np.array(read_csv(os.path.join(csv_dir, 'percentage_normal_final.csv')), dtype='float')
    percentage = np.concatenate((percentage_covid , percentage_normal), axis=0)
    percentage = percentage.tolist()
    
    slice_names_covid = read_csv(os.path.join(csv_dir, 'slice_level_names_covid_final.csv'))
    slice_names_normal = read_csv(os.path.join(csv_dir, 'slice_level_names_normal_final.csv'))
    slice_names = slice_names_covid + slice_names_normal
    
    slice_prob_covid = read_csv(os.path.join(csv_dir,'slice_level_prob_covid_final.csv'))
    slice_prob_normal = read_csv(os.path.join(csv_dir, 'slice_level_prob_normal_final.csv'))
    slice_prob = slice_prob_covid + slice_prob_normal
    
    labels = np.array([1] *(len(percentage_covid)) + [0] * (len(percentage_normal)))
    labels = np.reshape(labels, (-1,1))
    labels = labels.tolist()

    #Computing and drawing ROC curve 
    plt.figure(1)
    labelss = labels
    percentage_original = percentage.copy()
    pprob = percentage
    fpr, tpr, thresholds = roc_curve(labelss, pprob)
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(save_path,'roc_100.png'))
    # show the plot
    print(thresholds)
    print('--------------------------------------------------')
    
    # Computing and drawing Precision recall curve
    plt.figure(2)
    label = labelss
    lr_precision, lr_recall, thresholds = precision_recall_curve(label, pprob)
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(save_path,'precision_recall_curve100.png'))
    print(thresholds)
    
    # Computing metrics based on different thresholds
    total_acc = []
    total_precision = [] 
    total_recall = []
    total_f1_score = []
    total_suspicious = []
    pprob = percentage_original
    #suspicious = 0
    i = 0
    for t1 in np.arange(th1, th2, 0.01).tolist():
        for t2 in np.arange(th1, th2, 0.01).tolist():
          thre2, thre1 = sorted([t1, t2])
          i = -1
          pred = []
          label_2th = []
          suspicious = 0
          for p, l in zip(pprob, labelss): # each sample in current set
             i += 1
             if p[0] >= thre1:
        # if covid_confidance > th:
                  pred.append(1)  # 'covid'
                  label_2th.append(labelss[i])
             elif p[0] < thre2:
                  pred.append(0) # 'normal'
                  label_2th.append(labelss[i])
             else:
                  suspicious += 1
    
          #import pudb; pu.db
          acc = accuracy_score(label_2th, np.reshape(pred, (-1,1)))
          precision = precision_score(label_2th, np.reshape(pred, (-1,1)))
          recall = recall_score(label_2th, np.reshape(pred, (-1,1)))
          f1_scoret = f1_score(label_2th, np.reshape(pred, (-1,1)))
          confusion_matrixx = confusion_matrix(label_2th, np.reshape(pred, (-1,1)))
    
    
          total_acc.append(acc)
          total_precision.append(precision)
          total_recall.append(recall)
          total_f1_score.append(f1_scoret)
          total_suspicious.append(suspicious)
    
          #import pudb; pu.db
          print('threshold1: '+ str(thre1))
          print('threshold2: '+ str(thre2))
          print('accuracy: '+ str(acc))
          print('f1_score: '+ str(f1_scoret))
          print('precision: ' + str(precision))
          print('recall: ' + str(recall))
          print('# suspicious: ' + str(suspicious))
          print(confusion_matrixx)
          print('----------------------------------------------')
              
            
    plt.figure(3)        
    # metrics vs threshold
    th = np.arange(0, 1, 0.01).tolist()
    plt.plot(th, total_acc, 'r') # plotting t, a separately 
    plt.plot( th, total_precision, 'b') # plotting t, b separately 
    plt.plot(th, total_recall, 'g') # plotting t, c separately 
    plt.plot(th, total_f1_score, 'y')
    plt.legend(['acc', 'precision', 'recall', 'f1_score'])
    plt.savefig(os.path.join(save_path,'metrics100.png'))
    
    
