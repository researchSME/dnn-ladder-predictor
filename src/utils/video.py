import cv2
import subprocess
from shlex import split

def get_video_info(video_filename):
    cap = cv2.VideoCapture(video_filename)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame = cap.read()
    channels = frame.shape[2]
    cap.release()
    return width, height, channels, count

def get_video_frame_count(video_path: str):
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {video_path}"
    out = subprocess.check_output(split(cmd))
    return int(out)
