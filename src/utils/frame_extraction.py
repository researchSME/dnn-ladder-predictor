import subprocess
from shlex import split
from pathlib import Path
import os
import numpy as np
from utils.video import get_video_frame_count


def get_n_evenly_spaced_values_in_range(start: int, end: int, n: int):
    return list(np.linspace(start, end, 2+n, dtype=int)[1:-1])

def generate_output_frame_paths(output_dir: str, video_path: str, frame_nums: list):
    return  [f"{output_dir}/{Path(video_path).name}_frame{frame_num}.png" for frame_num in frame_nums]

def extract_frames_from_video(video_path: str, output_dir: str, frame_nums: list, out_width: int, out_height: int):
    output_paths = generate_output_frame_paths(output_dir, video_path, frame_nums)
    images_exist = [os.path.exists(output_path) for output_path in output_paths]

    if not all(images_exist):
        ffmpeg_select_filter = f'eq(n\,{frame_nums[0]})'
        for frame_num in frame_nums[1:]:
            ffmpeg_select_filter += f'+eq(n\,{frame_num})'
        cmd = f"ffmpeg -hide_banner -loglevel quiet  -i {video_path} " \
            f"-vf \"select='{ffmpeg_select_filter}',scale={out_width}x{out_height}:flags=lanczos\" -vsync 0 " \
            f"-frame_pts 1 -y {output_dir}/{Path(video_path).name}_frame%d.png"
        subprocess.check_output(split(cmd))
    return output_paths
    
def extract_middle_frame_from_video(video_path: str, output_dir: str):
    frame_count = get_video_frame_count(video_path)
    frame_num = int(frame_count/2)-1
    return extract_frames_from_video(video_path, output_dir, [frame_num])

def get_n_values_from_center_of_range(start: int, end: int, n: int):
    range_length = end-start+1
    if range_length > n:
        middle_point = start + np.ceil(range_length/2) - 1
        center_start = middle_point - np.floor((n-1)/2)
        center_end = middle_point + np.floor(n/2)
        return list(np.linspace(center_start, center_end, n, dtype=int))
    else:
        return list(np.linspace(start, end, range_length, dtype=int))
