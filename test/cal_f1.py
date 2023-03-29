from utils import *
import numpy as np




y = np.loadtxt('/raid/wangzihan/5th_ABAW/annotation/AU_Detection_Challenge/Aff_Wild2_val_label.txt')

pred = []
with open('/raid/wangzihan/5th_ABAW/test/val_result_1.txt') as f:
    data = f.readlines()

    for line in data:
        odom = line.strip().split(',')
        number_int = map(int,odom)
        pred.append(odom)
pred = np.array(pred,dtype=int)

print(pred)

batch_size = pred.shape[0]
class_nb = pred.shape[1]
statistics_list = []


for j in range(class_nb):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(batch_size):
        if pred[i][j] == 1:
            if y[i][j] == 1:
                TP += 1
            elif y[i][j] == 0:
                FP += 1

        elif pred[i][j] == 0:
            if y[i][j] == 1:
                FN += 1
            elif y[i][j] == 0:
                TN += 1
        else:
            assert False
    statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})


mean_f1_score, f1_score_list = calc_f1_score(statistics_list)

print(mean_f1_score)

print(f1_score_list)