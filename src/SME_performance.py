from ignite.metrics.metric import Metric
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
import math
import numbers

class SMEPerformance(Metric):
    def __init__(self, dataframe, resolution=1080):
        super(SMEPerformance, self).__init__()
        self.dataframe = dataframe
        self.resolution = resolution
    
    @staticmethod 
    def is_number(x):
        return isinstance(x, numbers.Number) and not math.isnan(x)
        
    def reset(self):
        self._relative_pred = []
        self._mapped_pred = []
        self._pred_crf = []
        self._pred_vmaf = []
        self._actual_crf  = []
        self._actual_vmaf = []
        self._all_pred_crf = []

    def update(self, output):
        y_pred, y = output
        actual_crf = y[0][0].item()
        scene_name = y[1][0]
        scene_number = y[2][0].item()
        pred_crf = y_pred[2][0].item()
        self._all_pred_crf.append(pred_crf)
        if self.is_number(actual_crf):
            vmaf_pred = self.dataframe[(self.dataframe['video_name'] == scene_name) &
                                    (self.dataframe['scene_number'] == scene_number) &
                                    (self.dataframe['height'] == self.resolution) &
                                    (self.dataframe['crf'] == np.round(pred_crf).astype(int))]['vmaf_mean'].item()
            
            vmaf_actual = self.dataframe[(self.dataframe['video_name'] == scene_name) &
                                        (self.dataframe['scene_number'] == scene_number) &
                                        (self.dataframe['height'] == self.resolution) &
                                        (self.dataframe['crf'] == np.round(actual_crf).astype(int))]['vmaf_mean'].item()
        
            self._actual_crf.append(actual_crf)
            self._actual_vmaf.append(vmaf_actual)
            self._relative_pred.append(y_pred[0][0].item())
            self._mapped_pred.append(y_pred[1][0].item())
            self._pred_crf.append(pred_crf)
            self._pred_vmaf.append(vmaf_pred)
        
        

    def compute(self):
        actual_crf = np.reshape(np.asarray(self._actual_crf), (-1,))
        relative_pred = np.reshape(np.asarray(self._relative_pred), (-1,))
        mapped_pred = np.reshape(np.asarray(self._mapped_pred), (-1,))
        pred_crf = np.reshape(np.asarray(self._pred_crf), (-1,))
        actual_vmaf = np.reshape(np.asarray(self._actual_vmaf), (-1,))
        pred_vmaf = np.reshape(np.asarray(self._pred_vmaf), (-1,))
        all_pred_crf = np.reshape(np.asarray(self._all_pred_crf), (-1,))
        
        
        SROCC = stats.spearmanr(actual_crf, relative_pred)[0]
        KROCC = stats.stats.kendalltau(actual_crf, relative_pred)[0]
        PLCC = stats.pearsonr(actual_crf, mapped_pred)[0]
        RMSE = np.sqrt(np.power(actual_crf-pred_crf, 2).mean())
        MAE = np.mean(np.abs(actual_crf-pred_crf))
        ACC = accuracy_score(actual_crf.astype(int), np.round(pred_crf).astype(int))
        ACC_1 = accuracy_n(actual_crf.astype(int), np.round(pred_crf).astype(int), 1)
        ACC_2 = accuracy_n(actual_crf.astype(int), np.round(pred_crf).astype(int), 2)
        VMAF_1 = accuracy_n(actual_vmaf, pred_vmaf, 1)
        VMAF_2 = accuracy_n(actual_vmaf, pred_vmaf, 2)
        VMAF_6 = accuracy_n(actual_vmaf, pred_vmaf, 6)
        VMAF_MAE =  np.mean(np.abs(actual_vmaf-pred_vmaf))
        return {'SROCC': SROCC,
                'KROCC': KROCC,
                'PLCC': PLCC,
                'RMSE': RMSE,
                'MAE': MAE,
                'ACC': ACC,
                'ACC_1': ACC_1,
                'ACC_2': ACC_2,
                'VMAF_1': VMAF_1,
                'VMAF_2': VMAF_2,
                'VMAF_6': VMAF_6,
                'VMAF_MAE': VMAF_MAE,
                'actual_crf': actual_crf,
                'relative_pred': relative_pred,
                'mapped_pred': mapped_pred,
                'pred_crf': pred_crf,
                'all_pred_crf': all_pred_crf
                }

def accuracy_n(y_true, y_pred, n):
    result = np.ndarray(y_true.shape, dtype=bool)
    for i, true_val in np.ndenumerate(y_true):
        if np.abs(true_val - y_pred[i]) <= n:
            result[i] = True
        else:
            result[i] = False
    return np.average(result)
