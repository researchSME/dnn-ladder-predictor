import os
import torch
import pandas as pd
import argparse
from torchvision import transforms
from torchvision import models as torch_models
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import random
from utils.frame_extraction import get_n_values_from_center_of_range
from utils.video import get_video_info
from time import time
from video_paths import get_video_paths
from utils.file_io import makedir


def get_transform(model_name, width=512, height=512):
    if model_name == "vgg16":
        return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif model_name == "inception_v3":
        return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif model_name == "resnet50":
        return transforms.Compose([transforms.Resize((height, width)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
def get_model(model_name):
    if model_name == "vgg16":
        return torch_models.vgg16(weights=torch_models.VGG16_Weights.DEFAULT)
    elif model_name == "inception_v3":
        return torch_models.inception_v3(weights=torch_models.Inception_V3_Weights.DEFAULT)
    elif model_name == "resnet50":
        return torch_models.resnet50(weights=torch_models.ResNet50_Weights.DEFAULT)

def gram(x):
    x = x.view(x.size()[0], x.size()[1], -1)
    return torch.matmul(x, torch.transpose(x, 1, 2))

def global_std_pool2d(x):
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)

class VideoDataset(Dataset):
    def __init__(self, dataframe, video_paths, transform, max_len=240):
        super(VideoDataset, self).__init__()
        self.dataframe = dataframe
        self.video_paths = video_paths
        self.width = transform.transforms[0].size[1]
        self.height = transform.transforms[0].size[0]
        self.max_frame_count = max_len
        self.transform = transform
    
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
        _, _, channel, _ = get_video_info(video_path)
        cap = cv2.VideoCapture(video_path)
        j = 0
        for _, scene in video_scenes.iterrows():
            scene_mid_frame_numbers = get_n_values_from_center_of_range(scene['start_frame'], scene['end_frame'], self.max_frame_count)
            scene_name = f"{video_name}_{str(int(scene['scene_number']))}"
            transformed_scene = torch.zeros([len(scene_mid_frame_numbers), channel,  self.height, self.width])
            cap.set(cv2.CAP_PROP_POS_FRAMES, scene_mid_frame_numbers[0])
            for i, _ in enumerate(scene_mid_frame_numbers):
                status, frame = cap.read()
                if not status:
                    print(f"{scene_name} read frame out of bound")
                    print(f"frame_number: {int(scene_mid_frame_numbers[0]) + i}")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                transformed_scene[i] = frame
            yield {'video_scene': transformed_scene,
                  'scene_name': scene_name}
            j += 1
    
class SpatialExtractor(torch.nn.Module):
    def __init__(self, model, layers=['layer4.2.relu_2'], pool_method='gram'):
        super(SpatialExtractor, self).__init__()
        self.pool_method = pool_method
        self.layers = layers
        self.feature_extractor = create_feature_extractor(model, {layer:layer for layer in layers})
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
    def forward(self, x):
        extracted_features = self.feature_extractor(x)
        pooled_keys = extracted_features.keys()
        if self.pool_method == 'gram':
            pooled_features = dict()
            for (pooled_layer, extracted_layer) in zip(sorted(pooled_keys), sorted(self.layers)):
                pooled_features[pooled_layer] = gram(extracted_features[extracted_layer]).flatten(1)
            return pooled_features
        else:
            pooled_features_mean = dict()
            pooled_features_std = dict()
            for (pooled_layer, extracted_layer) in zip(sorted(pooled_keys), sorted(self.layers)):
                curr_features = extracted_features[extracted_layer]
                pooled_features_mean[pooled_layer] = nn.functional.adaptive_avg_pool2d(curr_features, 1)
                pooled_features_std[pooled_layer] = global_std_pool2d(curr_features)
            return pooled_features_mean, pooled_features_std
            
def get_features(video_data, extractor, layers, frame_batch_size=64, device='cuda'):
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    if pool_method == 'gram':
        output = {layer:torch.Tensor().to(device) for layer in layers}
        extractor.eval()
        with torch.no_grad():
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end].to(device)
                features = extractor(batch)
                for layer in layers:
                    output[layer] = torch.cat((output[layer], features[layer]), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size
            last_batch = video_data[frame_start:video_length].to(device)
            features = extractor(last_batch)
            for layer in layers:
                    output[layer] = torch.cat((output[layer], features[layer]), 0)
    else:    
        output1 = {layer:torch.Tensor().to(device) for layer in layers}
        output2 = {layer:torch.Tensor().to(device) for layer in layers}
        output = dict()
        extractor.eval()
        with torch.no_grad():
            while frame_end < video_length:
                batch = video_data[frame_start:frame_end].to(device)
                features_mean, features_std = extractor(batch)
                for layer in layers:
                    output1[layer] = torch.cat((output1[layer], features_mean[layer]), 0)
                    output2[layer] = torch.cat((output2[layer], features_std[layer]), 0)
                frame_end += frame_batch_size
                frame_start += frame_batch_size
            last_batch = video_data[frame_start:video_length].to(device)
            features_mean, features_std = extractor(last_batch)
            for layer in layers:
                output1[layer] = torch.cat((output1[layer], features_mean[layer]), 0)
                output2[layer] = torch.cat((output2[layer], features_std[layer]), 0)
                output[layer] = torch.cat((output1[layer], output2[layer]), 1).squeeze()
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
    
    models = ['resnet50', 'vgg16', 'inception_v3']
    layers = {
        'resnet50':['layer4.2.relu_2'],
        'vgg16':['features.29'],
        'inception_v3':['Mixed_7c.cat_2']
        }
    pool_methods = ['mean_std']
    # pool_methods = ['mean_std', 'gram']
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_paths = get_video_paths(dataset_root_dir)
    dataframe = pd.read_csv(dataframe_path, index_col=[0], skipinitialspace=True)
    features_dir = os.path.join(features_dir_root, 'deep_features', 'spatial_features')
        
    for pool_method in pool_methods:
        print(f"pool_method: {pool_method}")
        for model_name in models:
            for layer in layers[model_name]:
                path = os.path.join(features_dir, model_name, str(layer), pool_method)
                makedir(path)
            print(f"model: {model_name}")
            print(f"layers: {layers[model_name]}")
            model = get_model(model_name)
            transform = get_transform(model_name, width, height)
            dataset = VideoDataset(dataframe, video_paths, transform, max_len=max_len)
            extractor = SpatialExtractor(model, layers[model_name], pool_method).to(device)
            for i in range(len(dataset)):
                start = time()
                skip = True
                for scene_name in dataset.get_scene_names(i):
                    for layer in layers[model_name]:
                        feature_path = os.path.join(features_dir, model_name, str(layer), pool_method, f'{scene_name}_spatial_features.npy')
                        if not os.path.exists(feature_path):
                            skip = False
                            break
                    if not skip:
                        break
                if not skip:
                    for current_video in dataset[i]:
                        scene_name = current_video['scene_name']
                        print(f"video: {scene_name}")
                        scene = current_video['video_scene']
                        features = get_features(scene, extractor, layers[model_name], frame_batch_size, device)
                        for layer in layers[model_name]:
                            feature_path = os.path.join(features_dir, model_name, str(layer), pool_method, f'{scene_name}_spatial_features.npy')
                            np.save(feature_path, features[layer].to('cpu').numpy())
                else:
                    print(f"video: {scene_name} skipped all scenes.")
                end = time()
                print(f"time to extract: {end - start}")
