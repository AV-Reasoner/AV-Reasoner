import io
import os
import base64
import json
import traceback
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

def process_item(item):
    output_path = f"long_acc/{item['index']}.txt"
    if os.path.exists(output_path):
        return

    video_path = f"cg_videos_720p/{item['video']}"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    prompt = f"Please answer the question '{item['question_en']}' with a number. Just output the number itself, don't output anything else."
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
        traceback.print_exc()
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