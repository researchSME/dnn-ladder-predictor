import os
import torch
import pandas as pd
import argparse
import numpy as np
from video_paths import get_video_paths
from utils.file_io import makedir

def fuse_features(all_features):
    min_feature_len = 100000
    for feature in all_features:
        min_feature_len = np.min((feature.shape[0], min_feature_len))
    fused_features = np.empty(shape=[min_feature_len, 0], dtype=np.float32)
    for feature in all_features:
        if feature.shape[0] != min_feature_len:
            index = torch.linspace(0, (feature.shape[0] - 1), min_feature_len).long()
            feature = torch.index_select(torch.tensor(feature), 0, index)
        fused_features = np.concatenate((fused_features, feature), 1)
    return fused_features

parser = argparse.ArgumentParser()
parser.add_argument('-dataframe_path', dest='dataframe_path', default='data/dataframes/videos_dataframe.csv', type=str, help="path to dataframe including video information")
parser.add_argument('-dataset_path', dest='dataset_root_dir', default='data/videos', type=str, help="root directory of video datasets")
parser.add_argument('-features_path', dest='features_dir_root', default='data/features', type=str, help="path to store extracted features")
if __name__ == "__main__":
    args = parser.parse_args()
    dataframe_path = args.dataframe_path
    features_dir_root = args.features_dir_root
    dataset_root_dir = args.dataset_root_dir
    
    deep_spatial_features_paths = {
        'resnet50': 'resnet50/layer4.2.relu_2/mean_std',
        'vgg16': 'vgg16/features.29/mean_std',
        'inception_v3': 'inception_v3/Mixed_7c.cat_2/mean_std'
        }
    deep_temporal_features_paths ={
        'slowfast': 'slowfast'
    }
    video_paths = get_video_paths(dataset_root_dir)
    dataframe = pd.read_csv(dataframe_path, index_col=[0], skipinitialspace=True)
    deep_spatial_features_permutation = [['resnet'], ['vgg16'], ['inception_v3'], ['resnet', 'vgg16'], ['resnet', 'inception_v3'], ['vgg16', 'inception_v3'], ['resnet', 'vgg16', 'inception_v3']]
    deep_temporal_features_permutation = [[], ['slowfast']]
    for deep_spatial_features in deep_spatial_features_permutation:
        for deep_temporal_features in deep_temporal_features_permutation:
            fuse_title = str()
            for feat in sorted(deep_spatial_features):
                feature_path = deep_spatial_features_paths[feat].replace('/', '_')
                fuse_title += f"{feature_path}__"
            for feat in sorted(deep_temporal_features):
                feature_path = deep_temporal_features_paths.replace('/', '_')
                fuse_title += f"{feature_path}__"
            fuse_title = fuse_title[:-2]
            if len(fuse_title) > 120:
                fuse_title = fuse_title[:120]
            fuse_dir = os.path.join(features_dir_root, 'fused_features', fuse_title)
            makedir(fuse_dir)
            max_feature_len = 0
            min_feature_len = 10000
            for j, scene in dataframe.iterrows():
                curr_features = list()
                video_name = scene['video_name']
                scene_number = str(int(scene['scene_number']))
                print(video_name, scene_number)
                for feat in deep_spatial_features:
                    curr_features.append(np.load(os.path.join(features_dir_root, 'deep_features', 'spatial_features', feat, f"{video_name}_{scene_number}_spatial_features.npy")))
                for feat in deep_temporal_features:
                    curr_features.append(np.load(os.path.join(features_dir_root, 'deep_features', 'temporal_features', feat, f"{video_name}_{scene_number}_temporal_features.npy")))
                features = fuse_features(curr_features)
                max_feature_len = np.max((features.shape[0], max_feature_len))
                min_feature_len = np.min((features.shape[0], min_feature_len))
                np.save(os.path.join(fuse_dir, f"{video_name}_{scene_number}_fused_features"), features)
            print(f"fuse_title: {fuse_title}")
            print(f"feature_size: {features.shape[1]}")
            print(f"max_feature_len: {max_feature_len}")
            print(f"min_feature_len: {min_feature_len}")
