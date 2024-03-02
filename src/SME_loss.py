import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsort
import numpy as np

class SMELoss(nn.Module):
    def __init__(self, scale, dataframe, loss_type='l1', min_label=None, resolution=1080):
        super(SMELoss, self).__init__()
        self.loss_type = loss_type
        self.scale = scale
        self.min_label = min_label
        self.dataframe = dataframe
        self.resolution = resolution

    def forward(self, y_pred, y):
        relative_score, mapped_score, aligned_score = y_pred
        y, scene_name, scene_number = y
        loss = 0
        if 'plcc' in self.loss_type:
            loss += loss_accuracy(mapped_score, y)
        if 'srcc' in self.loss_type:
            loss += loss_monotonicity(relative_score, y)
        if 'l1' in self.loss_type:
            loss += F.l1_loss(aligned_score, y) / self.scale
        if 'vmaf' in self.loss_type:
            vmaf_pred = torch.zeros_like(aligned_score)
            vmaf_actual = torch.zeros_like(aligned_score)
            for (i, pred), (j, actual) in zip(enumerate(aligned_score), enumerate(y)):
                vmaf_pred[i] = self.dataframe[(self.dataframe['video_name'] == scene_name[i]) &
                                              (self.dataframe['scene_number'] == scene_number[i].item()) &
                                              (self.dataframe['height'] == self.resolution) &
                                              (self.dataframe['crf'] == np.round(pred.item()).astype(int))]['vmaf_mean'].item()
                vmaf_actual[j] = self.dataframe[(self.dataframe['video_name'] == scene_name[i]) &
                                                (self.dataframe['scene_number'] == scene_number[i].item()) &
                                                (self.dataframe['height'] == self.resolution) &
                                                (self.dataframe['crf'] == np.round(actual.item()).astype(int))]['vmaf_mean'].item()
            loss += F.l1_loss(vmaf_pred, vmaf_actual) / 100
        return loss

def loss_accuracy(y_pred, y):
    """prediction accuracy related loss"""
    assert y_pred.size(0) > 1
    return (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2

def loss_monotonicity(y_pred, y, **kw):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1
    pred = torch.t(y_pred)
    pred = torchsort.soft_rank(pred, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = torch.t(y)
    target = torchsort.soft_rank(target, **kw)
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()
