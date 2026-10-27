import os,argparse, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix)
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.model_selection import train_test_split
from operator import truediv
import tensorflow as tf

## Different Dimensional Reduction Methods
def DLMethod(method, HSI, NC = 75):
    RHSI = np.reshape(HSI, (-1, HSI.shape[2]))
    if method == 'PCA': ## PCA
        pca = PCA(n_components = NC, whiten = True)
        RHSI = pca.fit_transform(RHSI)
        RHSI = np.reshape(RHSI, (HSI.shape[0], HSI.shape[1], NC))
    elif method == 'iPCA': ## Incremental PCA
        n_batches = 256
        inc_pca = IncrementalPCA(n_components = NC)
        for X_batch in np.array_split(RHSI, n_batches):
          inc_pca.partial_fit(X_batch)
        X_ipca = inc_pca.transform(RHSI)
        RHSI = np.reshape(X_ipca, (HSI.shape[0], HSI.shape[1], NC))
    return RHSI


def TrTeSplit(HSI, GT, trRatio, vrRatio, teRatio, randomState=345):
    # Split into train and test sets
    Tr, Te, TrC, TeC = train_test_split(HSI, GT, test_size=teRatio,
                                        random_state=randomState, stratify=GT)
    # Calculate the validation ratio based on the updated test and train ratios
    totalTrRatio = trRatio + vrRatio
    new_vrRatio = vrRatio / totalTrRatio
    # Split train set into train and validation sets
    Tr, Va, TrC, VaC = train_test_split(Tr, TrC, test_size=new_vrRatio,
                                        random_state=randomState, stratify=TrC)

    return Tr, Va, Te, TrC, VaC, TeC
## Prediction Model
# %%
## Prediction Model
def PreModel(Actual, model):
    '''
    모델 예측 함수. 모델을 사용하여 실제 데이터를 예측하고, 예측된 클래스 레이블을 반환
    클래스에서 0을 제외하고 예측된 클래스 레이블을 반환?
    Actual: np.array : actual data
    model: keras model : keras model
    '''
    ## Validation Prediction Model
    prediction = model.predict(Actual)  # forward pass
    argmax_prediction = np.argmax(prediction, axis=1) + 1  # Add 1 to the argmax result to shift the class labels
    non_zero_classes = np.unique(argmax_prediction)  # Get the unique non-zero class labels
    # Remove the 0 class label if it exists in the non_zero_classes array
    if 0 in non_zero_classes:
        non_zero_classes = non_zero_classes[non_zero_classes != 0]
    # Filter out the non-zero classes from the argmax_prediction array
    Pre = argmax_prediction[np.isin(argmax_prediction, non_zero_classes)]
    return Pre


## Assigning Class Labels for Final Classification and Confusion Matrices
def ClassificationReports(TeC, Te_Pred, target_names,zero_division=0):
    '''
    최종 분류 및 혼동 행렬을 위한 클래스 레이블 할당 함수
    TeC: np.array : test data
    Te_Pred: np.array : test prediction
    target_names: list : target names
    zero_division: int : zero division
    
    return
    classification: str : classification report
    confusion: np.array : confusion matrix
    oa: float : overall accuracy
    each_acc: np.array : each accuracy
    aa: float : average accuracy
    kappa: float : kappa coefficient
    '''
    Te_Pred = np.argmax(Te_Pred, axis=1)    # Get the argmax of the test prediction
    # scikit-learn classification report 생성 (precision, recall, F1-score etc.)
    classification = classification_report(np.argmax(TeC, axis=1), Te_Pred, target_names = target_names,zero_division=zero_division)
    oa = accuracy_score(np.argmax(TeC, axis=1), Te_Pred)    # Overall accuracy, OA
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred)   # Confusion matrix
    list_diag = np.diag(confusion) # Get the diagonal of the confusion matrix (각 클래스별 정분류 개수)
    list_raw_sum = np.sum(confusion, axis=1) # Get the sum of the rows of the confusion matrix (각 클래스별 실제 샘플 개수)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # Calculate the accuracy of each class 클래스별 정확도 계산 (정분류 / 샘플 개수), NaN처리
    aa = np.mean(each_acc)  # Average accuracy, AA
    kappa = cohen_kappa_score(np.argmax(TeC, axis=1), Te_Pred)  # Kappa coefficient
    return classification, confusion, oa*100, each_acc*100, aa*100, kappa*100   # Return the classification report, confusion matrix, OA, AA, and Kappa


## Writing Results in CSV files

def CSVResults(file_name, classification, confusion, Tr_Time, Te_Time, DL_Time, kappa, oa, aa, each_acc, memory_used=None):
    '''
    결과 csv 파일에 쓰기
    file_name: str : file name
    classification: str : classification report
    confusion: np.array : confusion matrix
    Tr_Time: float : training time
    Te_Time: float : test time
    DL_Time: float : dimensionality reduction time
    kappa: float : kappa coefficient
    oa: float : overall accuracy
    aa: float : average accuracy
    each_acc: np.array : each accuracy
    '''
    classification = str(classification)
    confusion = str(confusion)
    with open(file_name, 'w') as CSV_file:
      CSV_file.write('{} Tr_Time'.format(Tr_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Te_Time'.format(Te_Time))
      CSV_file.write('\n')
      CSV_file.write('{} DL_Time'.format(DL_Time))
      CSV_file.write('\n')
      CSV_file.write('{} Kappa accuracy (%)'.format(kappa))
      CSV_file.write('\n')
      CSV_file.write('{} Overall accuracy (%)'.format(oa))
      CSV_file.write('\n')
      CSV_file.write('{} Average accuracy (%)'.format(aa))
      CSV_file.write('\n')
      if memory_used is not None:
                    CSV_file.write('{} Inference Memory Usage (MB)\n'.format(round(memory_used, 2)))
                    CSV_file.write('\n')
      CSV_file.write('{}'.format(classification))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(each_acc))
      CSV_file.write('\n')
      CSV_file.write('{}'.format(confusion))
    return CSV_file


## Plot and Save Confusion Matrix
def Conf_Mat(Te_Pred, TeC, target_names):
    '''
    혼동 향렬 플롯 및 저장
    Te_Pred: np.array : test prediction
    TeC: np.array : test data
    target_names: list : target names
    '''
    plt.rcParams.update({'font.size': 12})
    Te_Pred = np.argmax(Te_Pred, axis=1)
    confusion = confusion_matrix(np.argmax(TeC, axis=1), Te_Pred, labels=np.unique(np.argmax(TeC, axis=1)))
    cm_sum = np.sum(confusion, axis=1, keepdims=True)
    cm_perc = confusion / cm_sum.astype(float) * 100
    annot = np.empty_like(confusion).astype(str)
    nrows, ncols = confusion.shape
    for l in range(nrows):
      for m in range(ncols):
        c = confusion[l, m]
        p = cm_perc[l, m]
        if l == m:
          s = cm_sum[l]
          annot[l, m] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
          annot[l, m] = ''
        else:
          annot[l, m] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(confusion, index=np.unique(target_names), columns=np.unique(target_names))
    return cm, annot
## Plot Ground Truths
def GT_Plot(CRDHSI, GT, model, WS, k):
    '''
    GT plot function?
    CRDHSI: np.array : data
    GT: np.array : ground truth
    model: keras model : keras model
    WS: int : window size
    k: int : number of bands
    '''
    Predicted = model.predict(CRDHSI)
    Predicted = np.argmax(Predicted, axis=1)
    height, width = np.shape(GT)
    ## Calculate the predicted Ground Truths
    outputs = np.zeros((height, width))
    count = 0
    for AA in range(height):
        for BB in range(width):
            target = int(GT[AA,BB])
            if target == 0:
                continue
            else:
                outputs[AA][BB] = Predicted[count]
                count = count+1
    return outputs