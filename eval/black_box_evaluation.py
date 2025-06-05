import json
import math
from pathlib import Path
import re
from word2number import w2n

txt_files = [f for f in Path("YOUR_OUTPUT_PATH_HERE").iterdir() if f.suffix == '.txt']

count = 0
correct = 0
correct_1 = 0
mae = 0
rmse = 0


for filename in txt_files:
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            try:
                it = json.loads(line.strip())
            except:
                continue

            try: # 部分模型输出的是文字，转成数字便于计算
                it["pred"] = w2n.word_to_num(it["pred"]) 
            except:
                it["pred"] = 0

            count += 1

            if abs(float(it["gt"]) - float(it["pred"])) < 1e-5: # 正确的
                correct += 1
            
            if abs(float(it["gt"]) - float(it["pred"])) <= 1:
                correct_1 += 1

            if abs(float(it["gt"]) - float(it["pred"])) <= max(2*float(it["gt"]),100): # 防止模型输出类似于10000000……的超长数字，导致MAE和RMSE严重失真，因此进行截断
                mae += abs(float(it["gt"]) - float(it["pred"]))
                rmse +=abs(float(it["gt"]) - float(it["pred"]))**2
            else:
                mae += abs(float(it["gt"])*2)
                rmse +=abs(float(it["gt"])*2)**2
            

print(f"count:{count}\n")
print(f"acc:{correct / count}\n")
print(f"off-by-one acc:{correct_1 / count}\n")
print(f"mae: {mae / count}\n")
print(f"rmse: {math.sqrt(rmse / count)}\n")