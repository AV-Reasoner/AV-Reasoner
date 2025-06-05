import glob
import cv2
import numpy as np
import pandas as pd
import json

from tqdm import trange


label2idx = {
    "background": 1,
    "accordion": 2,
    "airplane": 3,
    "axe": 4,
    "baby": 5,
    "bassoon": 6,
    "bell": 7,
    "bird": 8,
    "boat": 9,
    "boy": 10,
    "bus": 11,
    "car": 12,
    "cat": 13,
    "cello": 14,
    "clarinet": 15,
    "clipper": 16,
    "clock": 17,
    "dog": 18,
    "donkey": 19,
    "drum": 20,
    "duck": 21,
    "elephant": 22,
    "emergency-car": 23,
    "erhu": 24,
    "flute": 25,
    "frying-food": 26,
    "girl": 27,
    "goose": 28,
    "guitar": 29,
    "gun": 30,
    "guzheng": 31,
    "hair-dryer": 32,
    "handpan": 33,
    "harmonica": 34,
    "harp": 35,
    "helicopter": 36,
    "hen": 37,
    "horse": 38,
    "keyboard": 39,
    "leopard": 40,
    "lion": 41,
    "man": 42,
    "marimba": 43,
    "missile-rocket": 44,
    "motorcycle": 45,
    "mower": 46,
    "parrot": 47,
    "piano": 48,
    "pig": 49,
    "pipa": 50,
    "saw": 51,
    "saxophone": 52,
    "sheep": 53,
    "sitar": 54,
    "sorna": 55,
    "squirrel": 56,
    "tabla": 57,
    "tank": 58,
    "tiger": 59,
    "tractor": 60,
    "train": 61,
    "trombone": 62,
    "truck": 63,
    "trumpet": 64,
    "tuba": 65,
    "ukulele": 66,
    "utv": 67,
    "vacuum-cleaner": 68,
    "violin": 69,
    "wolf": 70,
    "woman": 71
}
mapping = {
    1: 'first',
    2: 'second',
    3: 'third',
    4: 'fourth',
    5: 'fifth',
    6: 'sixth',
    7: 'seventh',
    8: 'eighth',
    9: 'ninth',
    10: 'tenth'
}

def binary_mask_to_bbox(mask):
    # 找出 mask 中为 1 的坐标位置
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None  # mask 全为 0，没有目标

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 加 1 是为了得到封闭的框
    return [int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)]


idx2label = {value:key for key,value in label2idx.items()}

# 获取所有文件路径
df = pd.read_csv("AVSS/metadata.csv")
_idx = 0
output_path = "data/arig_train_rft.json"
data = []

for idx in trange(len(df)):
    line = df.iloc[idx]
    if line["split"] != "train":
        continue
    files = glob.glob(f"AVSS/{line['label']}/{line['uid']}/labels_semantic/*.png")
    for file in files:
        # 从路径中提取文件信息
        f = file.split("/")[-1].replace(".png",".jpg")
        index = int(file.split("/")[-1].replace(".png",""))
        audio = f"AVSS/{line['label']}/{line['uid']}/audio.wav"
        image_file = f"AVSS/{line['label']}/{line['uid']}/frames/{index}.jpg"

        item = []
        # 读取PNG图像并转换为灰度图像
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        height, width = image.shape
        # 将图像二值化（假设白色为前景）
        # 由于每个标签通常是一个不同的整数值，使用cv2.findContours时需要对每个标签值进行处理
        contours_per_label = {}

        # 获取所有标签的唯一值
        labels = np.unique(image)
        for label in labels:
            if label ==0:
                continue
            binary_mask = np.uint8(image == label)
            x_min,y_min,x_max,y_max = binary_mask_to_bbox(binary_mask)


            item.append({"coordinates":[[x_min,y_min],[x_max,y_max]],"name":idx2label[label]})

        record = {
            "image": image_file,
            "audio": audio,
            "id": f"avss_det_{_idx}",
            "task": "avss_det",
            "prompt":"<speech><image>\n" + f"""Based on the given image in size of {width}x{height} and its corresponding audio, please recognize the category of object making sound in the image, and then find out the bounding box coordinates of the object that makes the sound at the {mapping[index+1]} second of the audio. Possible object categories are: {'; '.join(label2idx.keys())}. Output the bounding box contours in the following JSON Format:\n[{{"coordinates": [[x_min,y_min],[x_max,y_max]],"name":"xxx"}},…]\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:
<think> ... </think> <answer> [{{"coordinates": [[x_min,y_min],[x_max,y_max]],"name":"xxx"}},…] </answer>\nPlease strictly follow the format.""",
            "solution": item,
        }
        data.append(record)
        _idx+=1
        


json.dump(data,open(output_path,"w"),indent=2)