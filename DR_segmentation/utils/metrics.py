import os
import sys
from tqdm import tqdm

import argparse
import logging
import time
import numpy as np
import torch
from cenet import CE_Net_
from utils.losses import BCELoss
from dataset import OrganData, get_coutour_embeddings, get_background_embeddings

from monai import metrics
from PIL import Image

def compute_performance(score_output, binary_output, label, metric, prefix=None, reduction='mean'):

    binary_output = torch.from_numpy(binary_output)
    label = torch.from_numpy(label)

    result = {}
    if 'confusion' in metric:
        confusion_matrix = metrics.get_confusion_matrix(binary_output, label, include_background=True)
        recall = metrics.compute_confusion_matrix_metric("sensitivity", confusion_matrix)
        precision = metrics.compute_confusion_matrix_metric("precision", confusion_matrix)
        f1_score = metrics.compute_confusion_matrix_metric("f1 score", confusion_matrix)
        accuracy = metrics.compute_confusion_matrix_metric("accuracy", confusion_matrix)

        result['recall'] = recall
        result['precision'] = precision
        result['f1'] = f1_score
        result['acc'] = accuracy


    if 'IoU' in metric:
        IoU = metrics.compute_iou(binary_output, label)
        result['IoU'] = IoU

    if 'AUC' in metric:
        TP = []
        FP = []
        FN = []
        TN = []
        score_output = score_output*255
        
        for threshold in tqdm(range(0,255)):
            temp_TP=0.0
            temp_FP=0.0
            temp_FN=0.0
            temp_TN=0.0

            for index in range(label.size(0)):
                gt_img=label[index,:,:,:]
                prob_img=score_output[index,:,:,:]

                # prob_img = prob_img.numpy()
                gt_img = gt_img.numpy()

                prob_img=(prob_img>=threshold)*1

                temp_TP = temp_TP + (np.sum(prob_img * gt_img))
                temp_FP = temp_FP + np.sum(prob_img * ((1 - gt_img)))
                temp_FN = temp_FN + np.sum(((1 - prob_img)) * ((gt_img)))
                temp_TN = temp_TN + np.sum(((1 - prob_img)) * (1 - gt_img))

            TP.append(temp_TP)
            FP.append(temp_FP)
            FN.append(temp_FN)
            TN.append(temp_TN)

        TP = np.asarray(TP).astype('float32')
        FP = np.asarray(FP).astype('float32')
        FN = np.asarray(FN).astype('float32')
        TN = np.asarray(TN).astype('float32')

        FPR = (FP) / (FP + TN)
        TPR = (TP) / (TP + FN)
        AUC = np.round(np.sum((TPR[1:] + TPR[:-1]) * (FPR[:-1] - FPR[1:])) / 2., 4)
        result['AUC'] = AUC


    if prefix:
        result = {prefix+'_'+key:value for key, value in result.items()}

    return result