import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
from video_split import get_train_video_names, get_test_video_names, get_val_video_names

class SMEDataset(Dataset):
    def __init__(self, dataframe, max_len, feature_size, features_dir, label_name='1080_top_quality_crf'):
        self.num_of_data = len(dataframe)
        self.features = np.zeros((self.num_of_data, max_len, feature_size), dtype=np.float32)
        self.labels = np.zeros((self.num_of_data, 1), dtype=np.float32)
        self.length = np.zeros(self.num_of_data, dtype=int)
        self.scene_names = list()
        self.scene_numbers = list()
        i = 0
        for _, scene in dataframe.iterrows():
            features = np.load(os.path.join(features_dir, f"{scene['video_name']}_{scene['scene_number']}_fused_features.npy"))
            self.features[i, :min(features.shape[0], max_len), :] = features[:min(features.shape[0], max_len), :]
            self.labels[i] = scene[label_name]
            self.length[i] = features.shape[0]
            self.scene_names.append(scene['video_name'])
            self.scene_numbers.append(scene['scene_number'])
            i += 1
            
    def __len__(self):
        return self.num_of_data

    def __getitem__(self, idx):
        return (self.features[idx], self.length[idx]), (self.labels[idx], self.scene_names[idx], self.scene_numbers[idx])
    
    
def get_data_loaders(dataframe, max_len=240, feature_size=4608, seed=1211234, label_name='1080_top_quality_crf',
                    features_dir='deep_features/fused_features', test_portion=0.20, val_portion=0.12,
                     batch_size=64, exp_id=0, inference=False, do_random_split=False):
    if do_random_split:
        split_seed = seed+exp_id*50
        train_data, test_data = train_test_split(
            dataframe, test_size=test_portion, random_state=(split_seed))
        train_data, val_data = train_test_split(
            train_data, test_size=1-((1-(val_portion+test_portion))/(1-test_portion)), random_state=(split_seed))
    else:
        train_video_names = get_train_video_names()
        test_video_names = get_test_video_names()
        val_video_names = get_val_video_names()
        vid_names = dataframe['video_name_and_scene_number']
        idxes = vid_names[vid_names.isin(train_video_names)].index
        train_data = dataframe.loc[idxes]
        idxes = vid_names[vid_names.isin(test_video_names)].index
        test_data = dataframe.loc[idxes]
        idxes = vid_names[vid_names.isin(val_video_names)].index
        val_data = dataframe.loc[idxes]
        
    train_data.dropna(subset=label_name, axis=0, inplace=True)
    train_data = train_data[train_data[label_name]>10]
    if not inference:
        test_data.dropna(subset=label_name, axis=0, inplace=True)
        test_data = test_data[test_data[label_name]>10]
    val_data.dropna(subset=label_name, axis=0, inplace=True)
    val_data = val_data[val_data[label_name]>10]
    labels = dataframe[label_name]
    min_label = labels.min()
    max_label = labels.max()
    scale = max_label - min_label
    train_dataset = SMEDataset(train_data, max_len, feature_size, features_dir, label_name=label_name)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False)

    val_dataset = SMEDataset(val_data, max_len, feature_size, features_dir, label_name=label_name)
    val_loader = torch.utils.data.DataLoader(val_dataset)

    test_dataset = SMEDataset(test_data, max_len, feature_size, features_dir, label_name=label_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, scale, min_label, test_data
