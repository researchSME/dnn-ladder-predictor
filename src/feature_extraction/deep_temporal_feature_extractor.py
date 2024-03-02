import os
import torch
import pandas as pd
import argparse
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from pathlib import Path
import numpy as np
import random
from time import time
from slowfast.visualization.utils import process_cv2_inputs
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
from slowfast.utils import logging
from slowfast.models import build_model
from utils.video import get_video_info
from utils.frame_extraction import get_n_values_from_center_of_range
from video_paths import get_video_paths
from utils.file_io import makedir

logger = logging.get_logger(__name__)

def make_motion_model():
    args = parse_args()
    cfg = load_config(args, cfg_path='data/config/SLOWFAST_8x8_R50.yaml')
    model = build_model(cfg, gpu_id=None)
    cu.load_test_checkpoint(cfg, model)
    return model

class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, dataframe, video_paths, width=None, height=None, max_len=240):
        super(VideoDataset, self).__init__()
        self.dataframe = dataframe
        self.video_paths = video_paths
        self.width = width
        self.height = height
        self.max_frame_count = max_len
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
    def get_scene_names(self, idx):
        video_path = self.video_paths[idx]
        video_name = Path(video_path).stem
        video_scenes = self.dataframe[self.dataframe['video_name'] == video_name].sort_values(by='scene_number', ascending=True)
        for _, scene in video_scenes.iterrows():
            yield  f"{video_name}_{str(int(scene['scene_number']))}"
        
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = Path(video_path).stem
        video_scenes = self.dataframe[self.dataframe['video_name'] == video_name].sort_values(by='scene_number', ascending=True)
        width, height, channel, _ = get_video_info(video_path)
        do_resize = not(self.width == width and self.height == height)
        cap = cv2.VideoCapture(video_path)
        j = 0
        for _, scene in video_scenes.iterrows():
            scene_mid_frame_numbers = get_n_values_from_center_of_range(scene['start_frame'], scene['end_frame'], self.max_frame_count)
            scene_name = f"{video_name}_{str(int(scene['scene_number']))}"
            transformed_scene = np.ndarray((len(scene_mid_frame_numbers), self.height, self.width, channel), dtype=int)
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_mid_frame_numbers[0])
            for i, _ in enumerate(scene_mid_frame_numbers):
                status, frame = cap.read()
                if not status:
                    print(f"{scene_name} read frame out of bound")
                    print(f"frame_number: {int(scene_mid_frame_numbers[0]) + i}")
                    break
                if do_resize:
                    frame = cv2.resize(frame, (self.height,self.width), interpolation=cv2.INTER_LANCZOS4)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                transformed_scene[i] = frame
            yield {'video_scene': transformed_scene,
                  'scene_name': scene_name}
            j += 1
    
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        model = make_motion_model()
        self.features = model

    def forward(self, x):
        x = self.features(x)
        features_mean = nn.functional.adaptive_avg_pool2d(x[1], 1)
        features_std = global_std_pool3d(x[1])
        features_mean = torch.squeeze(features_mean).permute(1, 0)
        features_std = torch.squeeze(features_std).permute(1, 0)
        return features_mean, features_std

def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1), dim=2, keepdim=True)

def global_std_pool3d(x):
    """3D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], x.size()[2], -1, 1), dim=3, keepdim=True)


def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    extractor = CNNModel().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        args = parse_args()
        cfg = load_config(args, cfg_path='data/config/SLOWFAST_8x8_R50.yaml')
        if video_length <= frame_batch_size:
            batch = video_data[0:video_length]
            inputs = process_cv2_inputs(batch, cfg)
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device)

            features_mean, features_std = extractor(inputs)
            output1 = torch.cat((output1, features_mean), 0)
            output2 = torch.cat((output2, features_std), 0)
            output = torch.cat((output1, output2), 1).squeeze()
        else:
            num_block = 0
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end]
                inputs = process_cv2_inputs(batch, cfg)
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].to(device)

                features_mean, features_std = extractor(inputs)
                output1 = torch.cat((output1, features_mean), 0)
                output2 = torch.cat((output2, features_std), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size
                num_block = num_block + 1

            last_batch = video_data[(video_length - frame_batch_size):video_length]
            inputs = process_cv2_inputs(last_batch, cfg)
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device)

            features_mean, features_std = extractor(inputs)
            index = torch.linspace(0, (frame_batch_size - 1), 32).long()
            last_batch_index = (video_length - frame_batch_size) + index
            elements = torch.where(last_batch_index >= frame_batch_size * num_block)
            output1 = torch.cat((output1, features_mean[elements[0], :]), 0)
            output2 = torch.cat((output2, features_std[elements[0], :]), 0)
            output = torch.cat((output1, output2), 1).squeeze()
    if output.ndim == 1:
        output = output.unsqueeze(0)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('-dataframe_path', dest='dataframe_path', default='data/dataframes/videos_dataframe.csv', type=str, help="path to dataframe including video information")
parser.add_argument('-dataset_path', dest='dataset_root_dir', default='data/videos', type=str, help="root directory of video datasets")
parser.add_argument('-features_path', dest='features_dir_root', default='data/features', type=str, help="path to store extracted features")
parser.add_argument('-width', dest='width', default=512, type=int, help="width of videos for feature extraction")
parser.add_argument('-height', dest='height', default=512, type=int, help="height of videos for feature extraction")
parser.add_argument('-batch_size', dest='frame_batch_size', default=64, type=int, help="batch size for feature extraction")
parser.add_argument('-seed', dest='seed', default=19901116, type=int, help="seed for randomization")
parser.add_argument('-max_frames', dest='max_len', default=240, type=int, help="maximum number of frames in a scene when extracting features")

if __name__ == "__main__":    
    args = parser.parse_args()
    dataframe_path = args.dataframe_path
    features_dir_root = args.features_dir_root
    width = args.width
    height = args.height
    frame_batch_size = args.frame_batch_size
    seed = args.seed
    max_len=args.max_len
    dataset_root_dir = args.dataset_root_dir
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_paths = get_video_paths(dataset_root_dir)
    dataframe = pd.read_csv(dataframe_path, index_col=[0], skipinitialspace=True)
    dataset = VideoDataset(dataframe, video_paths, width=width, height=height, max_len=max_len)
    features_dir = os.path.join(features_dir_root, 'deep_features', 'temporal_features', 'slowfast')
    makedir(features_dir)
    for i in range(len(dataset)):
        start = time()
        skip = True
        for scene_name in dataset.get_scene_names(i):
            feature_path = os.path.join(features_dir, f'{scene_name}_temporal_features.npy')
            if not os.path.exists(feature_path):
                skip = False
                break
        if not skip:
            for current_video in dataset[i]:
                scene_name = current_video['scene_name']
                print(f"video: {scene_name}")
                feature_path = os.path.join(features_dir, f'{scene_name}_temporal_features.npy')
                scene = current_video['video_scene']
                features = get_features(scene, frame_batch_size, device)
                np.save(feature_path, features.to('cpu').numpy())
        else:
            print(f"video: {scene_name} skipped all scenes.")
        end = time()
        print(f"time to extract: {end - start}")
