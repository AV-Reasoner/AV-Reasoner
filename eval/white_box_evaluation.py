from scipy.optimize import linear_sum_assignment
import numpy as np
import traceback
import numpy as np
from collections import defaultdict

def time_str_to_seconds(time_str: str) -> float:
    time_str = time_str.strip()
    if '.' in time_str:
        time_main, milliseconds = time_str.split('.')
        milliseconds = float(f"0.{milliseconds}")
    else:
        time_main = time_str
        milliseconds = 0.0

    parts = list(map(int, time_main.split(":")))
    
    if len(parts) == 2:
        minutes, seconds = parts
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    return total_seconds + milliseconds

def extract_outer_json(text):
    stack = []
    start_idx = None
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                start_idx = i  # 最外层起点
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.dumps(json.loads(candidate))
                    except json.JSONDecodeError:
                        continue  # 尝试下一个 JSON 块
    return None

def compute_tiou(t1, t2):
    """Temporal IoU"""
    inter_start = max(t1[0], t2[0])
    inter_end = min(t1[1], t2[1])
    inter = max(0.0, inter_end - inter_start)
    union = max(t1[1], t2[1]) - min(t1[0], t2[0])
    return inter / union if union > 0 else 0.0

def compute_sIoU(box1, box2):
    """
    Complete IoU (sIoU) between two bounding boxes.
    Args:
        box1 (list or np.array): [x1, y1, x2, y2] of ground truth box
        box2 (list or np.array): [x1, y1, x2, y2] of predicted box
    
    Returns:
        IoU (float): The IoU score between the two boxes.
    """

    # Ensure the coordinates are ordered: [min_x, min_y, max_x, max_y]
    box1 = np.array([min(box1[0], box1[2]), min(box1[1], box1[3]),
                     max(box1[0], box1[2]), max(box1[1], box1[3])])
    box2 = np.array([min(box2[0], box2[2]), min(box2[1], box2[3]),
                     max(box2[0], box2[2]), max(box2[1], box2[3])])

    # Compute the intersection area
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Compute areas of the individual boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union area
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0

    return iou

def greedy_matching(gt_instances, pred_instances, iou_func):
    """Greedy matching based on maximum IoU"""
    unmatched_gt = set(range(len(gt_instances)))
    unmatched_pred = set(range(len(pred_instances)))
    matches = []

    while unmatched_gt and unmatched_pred:
        max_iou = -1
        best_match = None
        for gt_idx in unmatched_gt:
            for pred_idx in unmatched_pred:
                iou = iou_func(gt_instances[gt_idx], pred_instances[pred_idx])
                if iou > max_iou:
                    max_iou = iou
                    best_match = (gt_idx, pred_idx)

        if best_match:
            gt_idx, pred_idx = best_match
            matches.append((gt_idx, pred_idx))
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)

    return matches

def compute_cluster_pair_wcs(gt, pred, iou_type):
    if iou_type == 'tIoU':
        loc_sum = 0.0
        for g in gt:
            loc_sum += max([compute_tiou(g, p) for p in pred] or [0.0])
        loc_acc = loc_sum / len(gt) if gt else 0.0
        count_penalty = 1.0 - abs(len(pred) - len(gt)) / max(len(gt), 1)
        # count_penalty = 1.0
        return math.sqrt(loc_acc * max(0,count_penalty))

    elif iou_type == 'sIoU':
        # group by frame index
        from collections import defaultdict
        gt_by_f = defaultdict(list)
        pred_by_f = defaultdict(list)
        for f, box in gt:
            gt_by_f[f].append(box)
        for f, box in pred:
            pred_by_f[f].append(box)

        all_f = set(gt_by_f) | set(pred_by_f)
        wcs = 0.0
        for f in all_f:
            gt_f = gt_by_f.get(f, [])
            pred_f = pred_by_f.get(f, [])
            matches = greedy_matching(gt_f, pred_f, compute_sIoU)
            loc_sum = sum([compute_sIoU(gt_f[i], pred_f[j]) for i, j in matches])
            loc_acc = loc_sum / len(gt_f) if gt_f else 0.0
            count_penalty = 1.0 - abs(len(pred_f) - len(gt_f)) / max(len(gt_f), 1)
            # count_penalty = 1.0
            wcs += math.sqrt(loc_acc * max(0,count_penalty))
        return wcs / max(len(all_f), 1)

    else:
        raise ValueError("Unsupported iou_type")

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the time limit.")

def compute_wcs_unlabeled(gt_clusters, pred_clusters, iou_type='tIoU', timeout=10): # 主要是给attribute用的，但是object和event视作一个cluster也能用
    # Set the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the alarm to go off in 'timeout' seconds

    try:
        # Original function logic
        K = len(gt_clusters)
        M = len(pred_clusters)

        # Build cost matrix (we want max score → min cost)
        score_matrix = np.zeros((K, M))
        for i in range(K):
            for j in range(M):
                score_matrix[i, j] = compute_cluster_pair_wcs(gt_clusters[i], pred_clusters[j], iou_type)

        cost_matrix = -score_matrix  # maximize score → minimize cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_scores = [score_matrix[i, j] for i, j in zip(row_ind, col_ind)]
        unmatched_gt = K - len(row_ind)

        # WCS = average over gt clusters (including unmatched = 0)
        total_wcs = sum(matched_scores)
        return total_wcs / K

    except TimeoutException:
        print(gt_clusters,pred_clusters)
        print("Function execution exceeded the time limit.")
        return None  # or you can return some default value to indicate timeout

    finally:
        signal.alarm(0)  # Cancel the alarm after the function completes or times out

import numpy as np
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from sklearn import metrics
score_list = []
error = 0
txt_files = [f for f in Path("YOUR_OUTPUT_PATH_HERE").iterdir() if f.suffix == '.txt']
for filename in txt_files:
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            try:
                ret = json.loads(line.strip())
            except:
                continue
            j = None
            content_match = re.search(r"<answer>(.*?)</answer>", str(ret["pred"]), re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else str(ret["pred"]).strip()
            try:
                try:
                    j = json.loads(student_answer)
                except:
                    j = json.loads(extract_outer_json(student_answer))
            except:

                traceback.print_exc()
                pass
            score = None
            if "input" not in ret.keys():
                ret["input"] = ret["line"]
            if j is not None:
                try:
                    if ret['input']['category'] == 'event':
                        pred = []
                        for e in j:
                            
                            if type(e[0]) == str and type(e[1]) == str and ":" in e[0] and ":" in e[1]:
                                pred.append([time_str_to_seconds(e[0]),time_str_to_seconds(e[1])])
                            else:
                                pred.append([float(e[0].split(" ")[0]) if type(e[0]) == str else e[0],float(e[1].split(" ")[0]) if type(e[1]) == str else e[1]])
                        gt = []
                        for e in ret["input"]["clue"]:
                            gt.append([float(e['start']),float(e['end'])])
                            
                        score = compute_wcs_unlabeled([gt], [pred], "tIoU")
                        # print(score)
                    elif ret['input']['category'] == 'object':
                        gt = []
                        clue_timestamp_list = []
                        for clue in ret["input"]["clue"]:
                            if clue["timestamp"] not in clue_timestamp_list:
                                clue_timestamp_list.append(clue["timestamp"])
                        for clue in ret["input"]["clue"]:
                            gt.append((clue_timestamp_list.index(clue["timestamp"]), clue['bbox']))
                        pred = []
                        for key in j.keys():
                            if "Frame" not in key:
                                continue
                            idx = int(key.replace("Frame","")) -1
                            if len(j[key]) == 0:
                                continue
                            if type(j[key][0]) == list and len(j[key][0]) == 4:
                                for e in j[key]:
                                    if type(e) == list and len(e) == 4:
                                        pred.append((idx, e))
                            elif type(j[key][0]) == list and len(j[key][0]) == 2:
                                for ii in range(int(len(j[key])//2)):
                                    if type(j[key][ii*2]) == list and len(j[key][ii*2]) == 2 and type(j[key][ii*2+1]) == list and len(j[key][ii*2+1]) == 2:
                                        pred.append((idx, [j[key][ii*2][0],j[key][ii*2][1],j[key][ii*2+1][0],j[key][ii*2+1][1]]))
                        score = compute_wcs_unlabeled([gt], [pred], "sIoU")
                        # print(score)
                    elif ret['input']['category'] == 'attribute':
                        gt = []
                        clue_timestamp_list = []
                        for clue_ in ret["input"]["clue"]:
                            for clue in clue_:
                                if clue["timestamp"] not in clue_timestamp_list:
                                    clue_timestamp_list.append(clue["timestamp"])
                        for clue_ in ret["input"]["clue"]:
                            gt_ = []
                            for clue in clue_:
                                gt_.append((clue_timestamp_list.index(clue["timestamp"]), clue['bbox']))
                            gt.append(gt_)
                        pred = {}
                        for key in j.keys():
                            if "Frame" not in key:
                                continue
                            idx = int(key.replace("Frame","")) -1
                            for e in j[key]:
                                if e['label'] not in pred.keys():
                                    pred[e['label']] = []
                                if 'bbox' in e:
                                    if len(e['bbox']) == 4 and type(e['bbox']) == list:
                                        pred[e['label']].append((idx, e['bbox']))
                                if 'bbox_2d' in e:
                                    if len(e['bbox_2d']) == 4 and type(e['bbox_2d']) == list:
                                        pred[e['label']].append((idx, e['bbox_2d']))    
                        pred_list = [pred[key] for key in pred]                     
                        score = compute_wcs_unlabeled(gt, pred_list, "sIoU")
                except:
                    traceback.print_exc()
                    print(ret)
            if score is not None:
                score_list.append(score)
            else:
                score_list.append(0)
                error += 1

print("wcs", sum(score_list)/len(score_list) * 100)
print("ifa", 1-(error/len(score_list)))
                        