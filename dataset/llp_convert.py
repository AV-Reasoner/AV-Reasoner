import glob
import json
import os
import re
import av
import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from decord import VideoReader, cpu
import threading
from moviepy.editor import VideoFileClip

df = glob.glob("LLP/converted_label/*.txt")

output_path = "data/llp_train_rft.json"

# 提取音频
def extract_audio(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path, logger=None)
        duration = video.duration
        uniform_sampled_frames = np.linspace(0, video.fps * video.duration - 1, 64, dtype=int)
        video.close()
        return duration,uniform_sampled_frames,video.fps
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None


def preprocess(filename):
    event = []
    with open(filename) as f:
        anno = f.read()
    visual =  re.findall(r"<visual>(.*?)</visual>", anno, re.DOTALL)
    audio =  re.findall(r"<audio>(.*?)</audio>", anno, re.DOTALL)
    for e in visual:
        _e = e.split(",")
        if len(_e) == 1:
            continue
        event.append(_e[0])
    for e in audio:
        _e = e.split(",")
        if len(_e) == 1:
            continue
        event.append(_e[0])

    return event


def process_entry(filename,idx):
    with open(filename) as f:
        anno = f.read()
    vid = filename.split("/")[-1].replace(".txt","")
    try:
        visual =  re.findall(r"<visual>(.*?)</visual>", anno, re.DOTALL)
        audio =  re.findall(r"<audio>(.*?)</audio>", anno, re.DOTALL)
        video_path = f"LLP/video/{vid}.mp4"
        audio_path = f"LLP/audio/{vid}.mp3"
        duration,uniform_sampled_frames,fps = extract_audio(video_path, audio_path)

        if duration is None:
            return None

        item = []
        for e in visual:
            _e = e.split(",")
            if len(_e) == 1:
                continue
            event = _e[0]
            for interval in _e[1:]:
                start,end = interval.replace("(","").replace(")","").split()
                item.append({
                    "start": f"{float(start):.2f} seconds",
                    "end": f"{float(end):.2f} seconds",
                    "event": event,
                    "type": "visual"
                })

        for e in audio:
            _e = e.split(",")
            if len(_e) == 1:
                continue
            event = _e[0]
            for interval in _e[1:]:
                start,end = interval.replace("(","").replace(")","").split()
                item.append({
                    "start": f"{float(start):.2f} seconds",
                    "end": f"{float(end):.2f} seconds",
                    "event": event,
                    "type": "audio"
                })
        sample_text = ""
        for frame_idx in uniform_sampled_frames:
            sample_text += f"{frame_idx/fps:.2f} seconds, "

        record = {
            "video": video_path,
            "audio": audio_path,
            "solution": item,
            "task": "llp",
            "id": f"llp_{idx}",
            "prompt":"<speech><image>\n" + f"""From the {duration:.2f}-second video, 64 frames are sampled at these timestamps: {sample_text}. Please describe the events and their time ranges from the video. Output in the following JSON Format:\n[{{"start": "xx.xx seconds", "end": "xx.xx seconds", "event": "aaa","type":"audio/visual"}},…]. The possible events are: {event_str}.\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:
<think> ... </think> <answer> [{{"start": "xx.xx seconds", "end": "xx.xx seconds", "event": "aaa","type":"audio/visual"}},…] </answer>\nPlease strictly follow the format."""
        }
        return record
    except Exception as e:
        print(f"Error in processing entry: {e}")
        return None

event_list =[]
for idx, line in enumerate(df):
    event = preprocess(line)
    if event is None:
        continue
    else:
        for e in event:
            if e not in event_list:
                event_list.append(e)

event_str = "; ".join(event_list)
# 多线程处理
data = []
with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_entry, line.strip(), idx) for idx, line in enumerate(df)]
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result is not None:
            data.append(result)

# 保存到 JSON 文件
json.dump(data, open(output_path, "w"), indent=2)