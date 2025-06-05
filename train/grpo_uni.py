# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
import traceback
from typing import Optional

from datasets import load_dataset

from trainer import OLAGRPOUniTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import numpy as np

import json

def extract_outer_json(text):
    stack = []
    json_start = -1
    opening = {'{': '}', '[': ']'}
    closing = {'}': '{', ']': '['}

    for i, char in enumerate(text):
        if char in opening:
            if not stack:
                json_start = i  
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
                if not stack:
                    json_str = text[json_start:i+1] 
                    try:
                        parsed_json = json.loads(json_str)
                        return parsed_json 
                    except json.JSONDecodeError:
                        pass 
    return None

class ARIGGRPOReward:
    def __init__(self):
        """
        Reward based on IoU between multiple bounding boxes.
        """

    @staticmethod
    def compute_iou(gt_bbox, pred_bbox):
        """
        Compute 2D IoU between two bounding boxes.

        Args:
            gt_bbox (tuple): Ground truth bounding box as (x1, y1, x2, y2)
            pred_bbox (tuple): Predicted bounding box as (x1, y1, x2, y2)

        Returns:
            float: IoU score.
        """
        # Ground truth bbox (x1, y1, x2, y2)
        x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
        # Predicted bbox (x1, y1, x2, y2)
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

        # Compute intersection
        ix1 = max(x1_gt, x1_pred)
        iy1 = max(y1_gt, y1_pred)
        ix2 = min(x2_gt, x2_pred)
        iy2 = min(y2_gt, y2_pred)

        inter_width = max(0, ix2 - ix1)
        inter_height = max(0, iy2 - iy1)
        intersection_area = inter_width * inter_height

        # Compute union
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        union_area = gt_area + pred_area - intersection_area

        # Compute IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def group_by_object(self, data):
        """
        Group bounding boxes by object name and aggregate their intervals.
        """
        grouped = defaultdict(list)
        for item in data:
            key = item['object_name']
            grouped[key].append((item['x1'], item['y1'], item['x2'], item['y2']))
        return grouped

    def compute_reward(self, preds, refs):
        """
        Compute IoU reward for each object and average.

        Args:
            preds (list of dict): Predicted bounding boxes.
            refs (list of dict): Ground truth bounding boxes.

        Returns:
            float: Final reward score (average IoU).
        """
        if len(preds) == 0 and len(refs) == 0:
            return 1.0
        pred_groups = self.group_by_object(preds)
        ref_groups = self.group_by_object(refs)

        scores = []
        for object_name in ref_groups:
            gt_bboxes = ref_groups[object_name]
            pred_bboxes = pred_groups.get(object_name, [])
            if not gt_bboxes:
                continue

            # Calculate IoU for each gt_bbox with all predicted bboxes
            ious = [max([self.compute_iou(gt_bbox, pred) for pred in pred_bboxes], default=0.0)
                    for gt_bbox in gt_bboxes]
            scores.append(np.mean(ious))

        return float(np.mean(scores)) if scores else 0.0
    
class VTGGRPOReward:
    def __init__(self, ignore_type=False):
        """
        Reward based on multi-segment event-level temporal IoU.
        """
        self.ignore_type = ignore_type

    @staticmethod
    def merge_intervals(intervals):
        """
        Merge overlapping intervals.
        """
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlap
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    @staticmethod
    def compute_tiou(gt_intervals, pred_intervals):
        """
        Compute tIoU between two sets of segments.
        """
        if not gt_intervals and not pred_intervals:
            return 1.0  # Both empty = perfect match
        if not gt_intervals or not pred_intervals:
            return 0.0

        gt_merged = VTGGRPOReward.merge_intervals(gt_intervals)
        pred_merged = VTGGRPOReward.merge_intervals(pred_intervals)

        # Compute intersection
        inter = 0.0
        for gs, ge in gt_merged:
            for ps, pe in pred_merged:
                inter += max(0, min(ge, pe) - max(gs, ps))

        # Compute union
        all_intervals = gt_merged + pred_merged
        union = 0.0
        for s, e in VTGGRPOReward.merge_intervals(all_intervals):
            union += e - s

        return inter / union if union > 0 else 0.0

    def group_by_event(self, data):
        """
        Group segments by (type, event) or (all, event) and collect all intervals.
        """
        grouped = defaultdict(list)
        for item in data:
            key = (item['type'] if not self.ignore_type else 'all', item['event'])
            grouped[key].append((item['start'], item['end']))
        return grouped

    def compute_reward(self, preds, refs):
        """
        Compute multi-segment tIoU per event and average.
        """
        pred_groups = self.group_by_event(preds)
        ref_groups = self.group_by_event(refs)

        scores = []
        for key in ref_groups:
            gt_intervals = ref_groups[key]
            pred_intervals = pred_groups.get(key, [])
            tiou = self.compute_tiou(gt_intervals, pred_intervals)
            scores.append(tiou)

        return float(np.mean(scores)) if scores else 0.0

@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="videollama2", metadata={"help": "Model type selected in the list: "})
    model_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})
    tune_adapter_llm: bool = field(default=False)
    # Connector Arguments
    mm_projector_type: Optional[str] = field(default='linear')
    mm_projector_a_type: Optional[str] = field(default='linear')
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_mlp_adapter_a: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter_a: Optional[str] = field(default=None)
    # Vision tower Arguments
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Audio tower Arguments
    audio_tower: Optional[str] = field(default=None)
    tune_audio_tower: bool = field(default=False)
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy","format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "json file path"},
    )



def accuracy_reward_arig(completions, solution, **kwargs):
    rewarder = ARIGGRPOReward()
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            try:
                j = json.loads(student_answer.strip())
                pred = []
                gt = []
                for item in j:
                    if "coordinates" in item.keys() and "name" in item.keys():
                        try:
                            pred.append({
                                "object_name":item['name'],
                                "x1":int(item['coordinates'][0][0]),
                                "y1":int(item['coordinates'][0][1]),
                                "x2":int(item['coordinates'][1][0]),
                                "y2":int(item['coordinates'][1][1]),
                            })
                        except:
                            pass
                for item in sol:
                    try:
                        gt.append({
                            "object_name":item['name'],
                            "x1":int(item['coordinates'][0][0]),
                            "y1":int(item['coordinates'][0][1]),
                            "x2":int(item['coordinates'][1][0]),
                            "y2":int(item['coordinates'][1][1]),
                        })
                    except:
                        pass
                reward = rewarder.compute_reward(pred, gt)

            except:
                try:
                    j = json.loads(extract_outer_json(student_answer.strip()))
                    pred = []
                    gt = []
                    for item in j:
                        if "coordinates" in item.keys() and "name" in item.keys():
                            try:
                                pred.append({
                                    "object_name":item['name'],
                                    "x1":int(item['coordinates'][0][0]),
                                    "y1":int(item['coordinates'][0][1]),
                                    "x2":int(item['coordinates'][1][0]),
                                    "y2":int(item['coordinates'][1][1]),
                                })
                            except:
                                pass
                    for item in sol:
                        try:
                            gt.append({
                                "object_name":item['name'],
                                "x1":int(item['coordinates'][0][0]),
                                "y1":int(item['coordinates'][0][1]),
                                "x2":int(item['coordinates'][1][0]),
                                "y2":int(item['coordinates'][1][1]),
                            })
                        except:
                            pass
                    reward = rewarder.compute_reward(pred, gt)
                except:
                    reward = 0.0
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward*10)
    return rewards

def accuracy_reward_vtg(completions, solution, **kwargs):
    rewarder = VTGGRPOReward()
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            try:
                j = json.loads(student_answer.strip())
                pred = []
                gt = []
                for item in j:
                    if "start" in item.keys() and "end" in item.keys() and "event" in item.keys() and "type" in item.keys():
                        try:
                            pred.append({
                                "start":float(item["start"].split(" ")[0]),
                                "end":float(item["end"].split(" ")[0]),
                                "event":item["event"].lower(),
                                "type":item["type"].lower()
                            })
                        except:
                            pass
                for item in sol:
                    try:
                        gt.append({
                            "start":float(item["start"].split(" ")[0]),
                            "end":float(item["end"].split(" ")[0]),
                            "event":item["event"].lower(),
                            "type":item["type"].lower()
                        })
                    except:
                        pass
                reward = rewarder.compute_reward(pred, gt)

            except:
                try:
                    j = json.loads(extract_outer_json(student_answer.strip()))
                    pred = []
                    gt = []
                    for item in j:
                        if "start" in item.keys() and "end" in item.keys() and "event" in item.keys() and "type" in item.keys():
                            try:
                                pred.append({
                                    "start":float(item["start"].split(" ")[0]),
                                    "end":float(item["end"].split(" ")[0]),
                                    "event":item["event"].lower(),
                                    "type":item["type"].lower()
                                })
                            except:
                                pass
                    for item in sol:
                        try:
                            gt.append({
                                "start":float(item["start"].split(" ")[0]),
                                "end":float(item["end"].split(" ")[0]),
                                "event":item["event"].lower(),
                                "type":item["type"].lower()
                            })
                        except:
                            pass
                    reward = rewarder.compute_reward(pred, gt)
                except:
                    reward = 0.0
        except Exception:
            pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward*10)
    return rewards

def accuracy_reward_qa_mcq(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                if content_match:
                    student_answer = content_match.group(1).strip()
                    # HACK, if First letter is correct reward 1
                    # Compare the extracted answers
                    if student_answer == sol[0]["answer"]:
                        reward = 1.0
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward*10)
    return rewards

def accuracy_reward_count(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion for completion in completions]
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                if content_match:
                    student_answer = content_match.group(1).strip()
                    # HACK, if First letter is correct reward 1
                    # Compare the extracted answers
                    if int(sol[0]["answer"]) != 0:
                        reward =  1- min(1.0,(abs(int(student_answer) - int(sol[0]["answer"])) / int(sol[0]["answer"])))
                    else:
                        reward = 1- min(1.0,(abs(int(student_answer) - int(sol[0]["answer"])) / 1))
                else:
                    reward = 0.0
            except Exception:
                traceback.print_exc()
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward*10)
    return rewards

def format_reward_vtg(completions, **kwargs):
    """Reward function that checks if the completion has a specific format,
    and adjusts based on the presence of <confidence> and its validity (0 to 1)."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    task = kwargs["task"]
    completion_contents = [completion for completion in completions]
    rewards = []

    for content in completion_contents:
        reward = 0.0
        # Check if the completion matches the think-answer format
        match = re.match(pattern, content, re.DOTALL)

        if match:
            content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if content_match:
                student_answer = content_match.group(1).strip()
                try:
                    j = json.loads(student_answer.strip())
                    correct = 0
                    for item in j:
                        if "start" in item.keys() and "end" in item.keys() and "event" in item.keys() and "type" in item.keys():
                            correct += 1
                    reward = correct / len(j)
                except:
                    traceback.print_exc()
                    try:
                        j = json.loads(extract_outer_json(student_answer.strip()))
                        correct = 0
                        for item in j:
                            if "start" in item.keys() and "end" in item.keys() and "event" in item.keys() and "type" in item.keys():
                                correct += 1
                        reward = correct / len(j)
                        reward /= 2
                    except:
                        traceback.print_exc()
                        reward = 0.0
        rewards.append(reward)
    return rewards


def format_reward_arig(completions, **kwargs):
    """Reward function that checks if the completion has a specific format,
    and adjusts based on the presence of <confidence> and its validity (0 to 1)."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    task = kwargs["task"]
    completion_contents = [completion for completion in completions]
    rewards = []

    for content in completion_contents:
        reward = 0.0
        # Check if the completion matches the think-answer format
        match = re.match(pattern, content, re.DOTALL)

        if match:
            content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if content_match:
                student_answer = content_match.group(1).strip()
                try:
                    j = json.loads(student_answer.strip())
                    correct = 0
                    for item in j:
                        if "coordinates" in item.keys() and "name" in item.keys():
                            correct += 1
                    if len(j) != 0:
                        reward = correct / len(j)
                    else:
                        reward = 1.0
                except:
                    traceback.print_exc()
                    try:
                        j = json.loads(extract_outer_json(student_answer.strip()))
                        correct = 0
                        for item in j:
                            if "coordinates" in item.keys() and "name" in item.keys():
                                correct += 1
                        if len(j) != 0:
                            reward = correct / len(j)
                            reward /= 2
                        else:
                            reward = 0.5
                    except:
                        traceback.print_exc()
                        reward = 0.0
        rewards.append(reward)
    return rewards

def format_reward_qa_mcq(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward(completions, **kwargs):
    task = kwargs["task"][0]
    print(task)
    if task == "avss_det":
        return format_reward_arig(completions, **kwargs)
    elif task == "llp" or task == "ave" or task == "unav":
        return format_reward_vtg(completions, **kwargs)
    elif task == "avqa" or task == "music_avqa" or task =="dvd_count" or task == "RepCount":
        return format_reward_qa_mcq(completions, **kwargs)
    
def accuracy_reward(completions, solution, **kwargs):
    task = kwargs["task"][0]
    if task == "avss_det":
        return accuracy_reward_arig(completions, solution, **kwargs)
    elif task == "llp" or task == "ave" or task == "unav":
        return accuracy_reward_vtg(completions, solution, **kwargs)
    elif task == "avqa" or task == "music_avqa":
        return accuracy_reward_qa_mcq(completions, solution, **kwargs)
    elif task == "dvd_count" or task == "RepCount":
        return accuracy_reward_count(completions, solution, **kwargs)

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    def make_conversation(example):
        if "prompt" in example.keys():
            return {"prompt":example["prompt"]}
        else:
            return {"prompt":None}
    dataset = dataset.map(make_conversation)

    trainer_cls = OLAGRPOUniTrainer
    import bitsandbytes as bnb
    optimizer_cls = bnb.optim.AdamW8bit

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model_id=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels    )

    checkpoints = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]

    if checkpoints:
    # Train and push the model to the Hub
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args )
