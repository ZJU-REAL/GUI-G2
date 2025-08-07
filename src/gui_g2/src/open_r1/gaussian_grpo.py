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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import numpy as np
from scipy.stats import multivariate_normal
import re
import logging
import ast
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
from trainer import VLMGRPOTrainer
from vlm_modules import *

from trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
import logging
import torch.distributed as dist  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from filelock import FileLock
from open_r1.vlm_modules.qwen_module import Qwen2VLModule
from open_r1.vlm_modules.internvl_module import InvernVLModule

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        # q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0))
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["gaussian_point","gaussian_plane","format"],
        metadata={"help": "List of reward functions. Possible values: 'point', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["instruction"]},
                ],
            }
        # FIXME
        # This is only for Grounding task
        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.respectively, i.e.<think> reasoning process here </think><answer> [x1, y1, x2, y2]</answer>. "
        # QUESTION_TEMPLATE = "{Question} Output the final answer in <answer> </answer> tags, i.e. <answer>[x1, y1, x2, y2]</answer>. "
        # QUESTION_TEMPLATE = "{Question} The output should be only [x1,y1,x2,y2]."
        # full_prompt = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'.format(instruction)

        # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.respectively, i.e.<think> reasoning process here </think><answer> answer bbox_2d: [x1, y1, x2, y2], label: xxx </answer>. "
        def make_conversation_image(example):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'.format(example["instruction"])},
                        ],
                    },
                ],
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image_path' in example:
            image_path = example['image_path']
            # In case the image is not found
            attempts = 0
            max_attempts = 5
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
                attempts += 1
                if attempts==5:
                    break
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                print('--------------- No image found....... ---------------')
                pass
        else:
            image = None
        return {
            'image': image,
            'image_path': image_path,
            'problem': example['instruction'],
            'solution': example['abs_box'],
            'prompt': make_conversation_image(example)['prompt'],
        }

def gaussian_plane_reward(completions, solution, **kwargs):
    def g_plane_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        eps   = 1e-8
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        pred_width = pred_x2 - pred_x1
        pred_height = pred_y2 - pred_y1
        # pred_μ
        pred_mu = np.array([pred_center_x, pred_center_y])

        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        # gt_μ
        gt_mu = np.array([gt_center_x, gt_center_y])
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1

        # 1 sigma
        pred_sigma_x = pred_width * alpha
        pred_sigma_y = pred_height * alpha
        gt_sigma_x   = gt_width * alpha
        gt_sigma_y = gt_height * alpha

        pred_cov = np.array([[pred_sigma_x**2, 0], 
                            [0, pred_sigma_y**2]])
        
        # Σ2 (ground truth distribution covariance matrix)  
        gt_cov = np.array([[gt_sigma_x**2, 0], 
                        [0, gt_sigma_y**2]])
        
        sigma_avg = (pred_cov + gt_cov) / 2
        # 
        mu_diff = pred_mu - gt_mu
        
        # (1/8) * (μ1 - μ2)^T * Σ^(-1) * (μ1 - μ2)
        sigma_avg_inv = np.linalg.inv(sigma_avg + eps * np.eye(2))
        term1 = (1/8) * np.dot(mu_diff.T, np.dot(sigma_avg_inv, mu_diff))
        
        # (1/2) * ln(det(Σ) / sqrt(det(Σ1) * det(Σ2)))
        det_sigma_avg = np.linalg.det(sigma_avg)
        det_pred_cov = np.linalg.det(pred_cov)
        det_gt_cov = np.linalg.det(gt_cov)
        try:
            term2 = 0.5 * np.log(det_sigma_avg / (np.sqrt(det_pred_cov * det_gt_cov + eps)))
        except:
            return 0.0
        bhattacharyya_distance = term1 + term2

        # 转换为奖励
        plane_reward = np.exp(-bhattacharyya_distance)
        plane_reward = round(plane_reward,3)
        return plane_reward

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        content = content.split('assistant\n')[-1]
        bbox_match = re.search(bbox_pattern, content.strip(), re.DOTALL)
        try:
            if bbox_match:
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                sol = [float(num) for num in sol]
                reward = g_plane_reward(bbox, sol)
        except Exception:
            print(Exception, content, sol)
            pass  
        
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"\n---------------------------------------------------- RANK: {0}, Coverage reward: {reward} ----------------------------------------------------\n")
                f.write(f"Image Path: \n{kwargs.get('image_path', ['N/A'])[0]}\n")
                f.write(f"\nInstruction: \n{kwargs.get('problem', ['N/A'])[0]}\n")
                f.write(f"\nTrue prompt: \n{kwargs.get('prompt', ['N/A'])[0]}\n")
                f.write(f"Content: \n{content}\n")
                f.write(f"\nSolution: \n{sol}\n")
    return rewards


def gaussian_point_reward(completions, solution, **kwargs):
    def g_point_reward(pred_bbox, gt_bbox):
        alpha = 0.5
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
        
        # 计算中心点
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        gt_center_x = (gt_x1 + gt_x2) / 2
        gt_center_y = (gt_y1 + gt_y2) / 2
        gt_width = gt_x2 - gt_x1
        gt_height = gt_y2 - gt_y1
        
        sigma_x = alpha * gt_width
        sigma_y = alpha * gt_height

        x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
        y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
        exponent = -0.5 * (x_term + y_term)
        point_reward = math.exp(exponent)
        point_reward = round(point_reward,3)
        return point_reward

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        content = content.split('assistant\n')[-1]
        bbox_match = re.search(bbox_pattern, content.strip(), re.DOTALL)
        try:
            if bbox_match:
                bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)), float(bbox_match.group(3)), float(bbox_match.group(4))]
                sol = [float(num) for num in sol]
                reward = g_point_reward(bbox, sol)
        except Exception:
            print(Exception, content, sol)
            pass  
        
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"\n---------------------------------------------------- RANK: {0}, point reward: {reward} ----------------------------------------------------\n")
                f.write(f"Image Path: \n{kwargs.get('image_path', ['N/A'])[0]}\n")
                f.write(f"\nInstruction: \n{kwargs.get('problem', ['N/A'])[0]}\n")
                f.write(f"\nTrue prompt: \n{kwargs.get('prompt', ['N/A'])[0]}\n")
                f.write(f"Content: \n{content}\n")
                f.write(f"\nSolution: \n{sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
    completion_contents = [completion[0]["content"] for completion in completions]

    matches = [re.fullmatch(pattern, content.split('assistant\n')[-1], re.DOTALL) for content in completion_contents]
    for i, num in enumerate([1.0 if match else 0.0 for match in matches]):
        if num < 1:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a") as f:
                    f.write(f"\n|||||||||||||||||||||||||||||||||||||||||||||||||||| RANK: {dist.get_rank()}, match: {num} ||||||||||||||||||||||||||||||||||||||||||||||||||||\n")
                    f.write(f"Image Path: \n{kwargs['image_path'][0]}\n")
                    # f.write(f"Width: {kwargs['width'][0]}, Height: {kwargs['height'][0]}\n")
                    f.write(f"\nInstruction: \n{kwargs['problem'][0]}\n")
                    f.write(f"\nformat not matched\n")
                    f.write(f"completion_contents: \n{completion_contents[i]}\n")
    return [1.0 if match else 0.0 for match in matches]

def object_to_dict(obj):
    """
    将类实例的属性转换为字典。
    """
    return {key: value for key, value in obj.__dict__.items()}

def write_configs_to_txt(filename, *configs):
    """
    将多个配置字典写入 txt 文件。
    
    参数：
    - filename: 输出文件名。
    - configs: 多个配置字典。
    """
    with open(filename, 'a', encoding='utf-8') as f:
        for i, config in enumerate(configs):
            # 写入每个配置的标题
            if i == 0:
                f.write("=== GRPOScriptArguments ===\n")
            elif i == 1:
                f.write("\n=== GRPOConfig ===\n")
            elif i == 2:
                f.write("\n=== ModelConfig ===\n")
            
            # 写入配置内容
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

reward_funcs_registry = {
    "gaussian_point": gaussian_point_reward,
    "gaussian_plane": gaussian_plane_reward,
    "format": format_reward,
}

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")


def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    # dist.init_process_group(backend="nccl")
    # import rpdb; rpdb.set_trace(port=4444+int(dist.get_rank()))
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)
    print(script_args.max_pixels, script_args.min_pixels)

    # Load the dataset
    
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    trainer_cls = VLMGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        if dist.get_rank()==0:
            write_configs_to_txt(log_path, object_to_dict(script_args), object_to_dict(training_args), object_to_dict(model_args))

    main(script_args, training_args, model_args)
