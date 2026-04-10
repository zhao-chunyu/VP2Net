import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, recall_score, accuracy_score,
                             average_precision_score)


def str_2_list(str_tmp):
    start_index = str_tmp.find('[')
    end_index = str_tmp.find(']')
    str_content = str_tmp[start_index + 1:end_index]
    list_tmp = str_content.split(', ')
    list_res = [float(i) for i in list_tmp]
    return list_res


def metric_tool(res_name):
    file = open(res_name, mode='r')
    lines = file.readlines()
    test_cnt = math.floor(len(lines) / 2)

    y_ture = []
    y_pred = []
    for i in range(0, test_cnt):
        pred = str_2_list(lines[2 * i])
        y_pred.append(pred.index(max(pred)))
        target = str_2_list(lines[2 * i + 1])
        y_ture.append(target.index(max(target)))

    acc = accuracy_score(y_ture, y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro')

    pred_res = [[], [], [], [], [], []]
    target_res = [[], [], [], [], [], []]

    for class_index in range(6):
        for i in range(test_cnt):
            pred = str_2_list(lines[2 * i])
            pred_res[class_index].append(pred[class_index])
            target = str_2_list(lines[2 * i + 1])
            target_res[class_index].append(target[class_index])
    mAP = []
    for i in range(6):
        mAP.append(average_precision_score(target_res[i], pred_res[i]))

    map_val = np.mean(mAP)
    file.close()

    return acc, f1, map_val


def acc_per_class(res_name):
    file = open(res_name, mode='r')
    lines = file.readlines()
    test_cnt = math.floor(len(lines) / 2)

    y_ture = np.zeros(6)
    y_pred = np.zeros(6)

    for i in range(0, test_cnt):
        pred = str_2_list(lines[2 * i])
        target = str_2_list(lines[2 * i + 1])
        class_index = target.index(max(target))
        if class_index in [0, 1, 2, 3, 4, 5]:
            y_ture[class_index] += 1
            if class_index == pred.index(max(pred)):
                y_pred[class_index] += 1

    file.close()
    return y_ture, y_pred, y_pred / y_ture


def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()