from math import log10
import torch
import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
from scipy.special import softmax
import scipy.stats

from sklearn.metrics import roc_curve, auc
from scipy import interp

def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)

def one_hot(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx

# def auc(preds, labels, is_logit=True):
#     ''' input: logits, labels  '''
#     if is_logit:
#         pos_probs = softmax(preds, axis=1)[:, 1]
#     else:
#         pos_probs = preds[:,1]
#     try:
#         auc_out = roc_auc_score(labels, pos_probs)
#     except:
#         auc_out = 0
#     return auc_out


def auc_2(preds, labels, num_classes=2, is_logit=True):
    ''' input: logits, labels  '''
    if num_classes > 2:#np.max(labels) - np.min(labels) > 1
        # my_list = [0, 1, 2, 3]
        # if 3 in my_list:
        #     # 将第i个类别的标签设为1，其他类别的标签设为0
        #     y_true_i = (labels == 3).astype(int)
        #     # 获取第i个类别的预测概率
        #     y_pred_i = preds[:, 3]
        #     # 计算该类别的AUC
        #     auc = roc_auc_score(y_true_i, y_pred_i)
        # return auc

        return auc_m(preds, labels, num_classes, is_logit=True)

    else:
        if is_logit:
            pos_probs = softmax(preds, axis=1)[:, 1]
        else:
            pos_probs = preds[:, 1]
        try:
            auc_out = roc_auc_score(labels, pos_probs)
        except:
            auc_out = 0

        # num_samples = [np.sum(labels == 0), np.sum(labels == 1)]  # 标签数量分别为label==0和label==1的数量
        # # 计算每个类别的权重
        # weights = [1.0, num_samples[1] / num_samples[0]]
        # # 根据权重计算分类阈值
        # weights_sum = weights[0] + weights[1]
        # weights[0] /= weights_sum
        # weights[1] /= weights_sum
        # threshold = np.percentile(preds[:, 1], (1 - weights[1]) * 100)
        # # 根据分类阈值计算AUC
        # y_pred_binary = (preds[:, 1] >= threshold).astype(int)
        # auc_out = roc_auc_score(labels, y_pred_binary)

        return auc_out

def prf_2(preds, labels, num_classes=2, is_logit=True):
    ''' input: logits, labels  '''
    if num_classes > 2:#np.max(labels) - np.min(labels) > 1
        return prf_m(preds, labels, is_logit=True)
    else:
        pred_lab= np.argmax(preds, 1)
        # print(labels)
        # print(pred_lab)
        p,r,f,s  = precision_recall_fscore_support(labels, pred_lab, average='binary', zero_division=1)
        return [p,r,f]

def auc_m(preds, labels, num_classes=2, is_logit=True):
    ''' input: logits, labels  '''
    # n_classes = np.max(labels) + 1
    n_classes = num_classes

    # from sklearn.preprocessing import label_binarize
    # label_binarize(labels, np.arange(n_classes))
    y_label = np.zeros((len(labels), n_classes))
    y_label[range(len(labels)), labels.astype(np.int64)] = 1

    y_score = softmax(preds, axis=1)
    # each class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    auc_out = roc_auc["macro"]

    return auc_out

def prf_m(preds, labels, is_logit=True):
    ''' input: logits, labels  '''
    pred_lab = np.argmax(preds, 1)
    p, r, f, s = precision_recall_fscore_support(labels, pred_lab, average='weighted', zero_division=1)
    return [p, r, f]



