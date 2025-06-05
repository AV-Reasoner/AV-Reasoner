import io
import os
import base64
import json
import requests
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed


API_BASE = ''
API_KEY = ''
headers = {
    'Authorization': API_KEY,
    'Content-Type': 'application/json',
}



def get_frames_at_times(video_path, timestamps_sec):
    # 打开视频文件
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    frames = []
    # 获取视频的帧率
    fps = vr.get_avg_fps()

    for timestamp_sec in timestamps_sec:
        # 计算视频帧对应的索引
        frame_idx = int(timestamp_sec * fps)

        # 获取对应帧
        frame = vr[frame_idx]

        # 将帧转换为PIL图像
        img = Image.fromarray(frame.asnumpy())
        frames.append(img)

    return frames


def sample_frames_global_average(max_frame, num_segment):
    seg_size = float(max_frame) / num_segment
    frame_indices = np.array([
        int(seg_size / 2 + np.round(seg_size * idx))
        for idx in range(num_segment)
    ])
    return frame_indices

def load_video_pipeline(video_path, num_frames):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_indices = sample_frames_global_average(max_frame, num_frames)
    frames = vr.get_batch(frame_indices).asnumpy()
    pil_images = [Image.fromarray(frame).convert('RGB') for frame in frames]
    timestamps = [round(idx / fps, 2) for idx in frame_indices]

    return pil_images, timestamps

def inference(prompt, images, messages, timestamps=None):
    content = []
    content = [{'type': 'text', 'text': "This is a video:"}]
    for i, image in enumerate(images):
        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")
        encoded_string = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
        img_src_attr_value = f'data:image/jpeg;base64,{encoded_string}'
        if timestamps:
            content.append({'type': 'text', 'text': f'Frame{i+1} sampled at {timestamps[i]} seconds'})
        content.append({'type': 'image_url', 'image_url': {'url': img_src_attr_value, 'detail': 'low'}})
    content.append({'type': 'text', 'text': prompt})
    messages.append({'role': 'user', 'content': content})
    json_data = {
        'model': 'gpt-4.1-2025-04-14',
        'messages': messages,
        'stream': False,
        'temperature': 0.0,
        'top_p': 1.0
    }
    response = requests.post(API_BASE, headers=headers, json=json_data, timeout=600)
    return response.json(), messages

def chat_with_gpt(num_frames, video_path, prompt):
    images, timestamps = load_video_pipeline(video_path, num_frames)
    result, _ = inference(prompt, images, [], timestamps)
    return result

def inference_frame(prompt, images, messages, width,height):
    content = [{'type': 'text', 'text': f"There are {len(images)} frames in the size of {width}x{height}"}]
    for i, image in enumerate(images):
        byte_stream = io.BytesIO()
        image.save(byte_stream, format="JPEG")
        encoded_string = base64.b64encode(byte_stream.getvalue()).decode('utf-8')
        img_src_attr_value = f'data:image/jpeg;base64,{encoded_string}'
        content.append({'type': 'text', 'text': f'Frame{i+1}:'})
        content.append({'type': 'image_url', 'image_url': {'url': img_src_attr_value, 'detail': 'low'}})
    content.append({'type': 'text', 'text': prompt})
    messages.append({'role': 'user', 'content': content})
    json_data = {
        'model': 'gpt-4.1-2025-04-14',
        'messages': messages,
        'stream': False,
        'temperature': 0.0,
        'top_p': 1.0
    }
    response = requests.post(API_BASE, headers=headers, json=json_data, timeout=600)
    return response.json(), messages

def chat_with_gpt_frame(timestamps, video_path, prompt):
    images = get_frames_at_times(video_path,timestamps)
    width = images[0].width
    height = images[0].height
    result, _ = inference_frame(prompt, images, [], width,height)
    return result

def process_item(item):
    output_path = f"clue_acc/{item['index']}.txt"
    if os.path.exists(output_path):
        return

    video_path = f"cg_videos_720p/{item['video']}"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    if item["category"] == "event":
        prompt = f"Watch the video and provide your answer to the question '{item['question_en']}', including the start and end timestamps for each event. Format your answer in JSON, enclosed in <answer> and </answer> tags. The output should look like this: <answer>[[\"start_time\", \"end_time\"], ...]</answer>. Ensure each timestamp is in seconds (e.g., 'xx.xx')."
        try:
            result = chat_with_gpt(num_frames=50, video_path=video_path, prompt=prompt)
            if "choices" in result:
                with open(output_path, "w") as f:
                    f.write(json.dumps({
                        "pred": result['choices'][0]['message']['content'],
                        "gt": item["answer"],
                        "input": item
                    }))
                print(f"[✓] {item['index']} completed.")
            else:
                print(f"[✗] {item['index']} failed: No 'choices' in result"+result["error"]["message"])
        except Exception as e:
            print(f"[✗] {item['index']} exception: {str(e)}")
    elif item["category"] == "object":
        clue_timestamp_list = []
        for clue in item["clue"]:
            if clue["timestamp"] not in clue_timestamp_list:
                clue_timestamp_list.append(clue["timestamp"])
        prompt = f"Watch the video and answer the question '{item['question_en']}', including the bounding box for the query object in the first frame where it appears. For subsequent frames where the object appears, do not provide the bounding box again. Format your answer in JSON, enclosed within <answer> and </answer> tags. The output should look like this: <answer>{{\"Frame1\": [[x_min, y_min, x_max, y_max]], \"Frame2\": [...],...}}</answer>. In the output, each frame should either contain the bounding box of the object (if it appears for the first time in that frame) or an empty list `[]` (if the object does not appear or it has already been labeled in a previous frame). Ensure that bounding boxes are listed as [x_min, y_min, x_max, y_max]."
        try:
            result = chat_with_gpt_frame(timestamps=clue_timestamp_list, video_path=video_path, prompt=prompt)
            if "choices" in result:
                with open(output_path, "w") as f:
                    f.write(json.dumps({
                        "pred": result['choices'][0]['message']['content'],
                        "gt": item["answer"],
                        "input": item
                    }))
                print(f"[✓] {item['index']} completed.")
            else:
                print(f"[✗] {item['index']} failed: No 'choices' in result"+result["error"]["message"])
        
        except Exception as e:
            print(f"[✗] {item['index']} exception: {str(e)}")
    elif item["category"] == "attribute":
        clue_timestamp_list = []
        for clue_ in item["clue"]:
            for clue in clue_:
                if clue["timestamp"] not in clue_timestamp_list:
                    clue_timestamp_list.append(clue["timestamp"])
        
        prompt = f"Watch the video and answer the question '{item['question_en']}', clustering the objects according to the question. For each unique cluster, assign a unique label and return the bounding box for each object in the first frame where it appears. For subsequent frames where the object appears, do not output anything. Format your answer in JSON, enclosed within <answer> and </answer> tags. The output should look like this: <answer>{{\"Frame 1\": [{{\"bbox\": [x_min, y_min, x_max, y_max], 'label': \"Label 1\"}}], \"Frame 2\": [...], ...}}</answer>. In the output, each frame should either contain the bounding box and label for the object (if it appears for the first time in that frame) or an empty list `[]` (if the object has already been labeled or does not appear in that frame). The label should correspond to a unique object cluster according to the question."
        try:
            result = chat_with_gpt_frame(timestamps=clue_timestamp_list, video_path=video_path, prompt=prompt)
            if "choices" in result:
                with open(output_path, "w") as f:
                    f.write(json.dumps({
                        "pred": result['choices'][0]['message']['content'],
                        "gt": item["answer"],
                        "input": item
                    }))
                print(f"[✓] {item['index']} completed.")
            else:
                print(f"[✗] {item['index']} failed: No 'choices' in result"+result["error"]["message"])
        
        except Exception as e:
            print(f"[✗] {item['index']} exception: {str(e)}")

def main():
    data_path = "cg-av-counting.json"
    data = json.load(open(data_path))

    max_workers = 24
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        for future in as_completed(futures):
            _ = future.result()

if __name__ == "__main__":
    main()