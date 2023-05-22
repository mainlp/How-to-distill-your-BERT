# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
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

import json
import logging
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class Logger:
    def __init__(self, cuda=False):
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda

    def info(self, message, ignore_rank=False, *args, **kwargs):
        if (
            not ignore_rank and self.cuda and dist.is_initialized() and dist.get_rank() == 0
        ) or not self.cuda:
            self.logger.info(message, *args, **kwargs)
        if ignore_rank:
            self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warn(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)


def master_process(args):
    return dist.get_rank() == 0 or args.local_rank == -1


def get_time_diff_hours(now, start_marker):
    """return difference between 2 time markers in hours"""
    time_diff = now - start_marker
    time_diff_hours = time_diff / 3600
    return time_diff_hours


def is_time_to_exit(now, args, epoch_steps=0, global_steps=0):
    time_diff_hours = get_time_diff_hours(now, args.exp_start_marker)

    # if passed max_pretrain_hours, then exit
    if time_diff_hours > args.total_training_time or time_diff_hours > args.early_exit_time_marker:
        return True

    return (epoch_steps >= args.max_steps_per_epoch) or (global_steps >= args.max_steps)


def is_time_to_finetune(now, start_marker, time_markers, total_time):
    if time_markers is None:
        return False
    time_diff_hours = get_time_diff_hours(now, start_marker)
    if len(time_markers) > 0 and time_diff_hours / total_time > time_markers[0]:
        time_markers.pop(0)
        return True
    else:
        return False


def get_json_file(path):
    with open(path, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)


def to_sanitized_dict(args) -> Dict[str, Any]:
    """Sanitized serialization hparams"""
    if type(args) in [dict]:
        d = args
    else:
        d = vars(args)
    valid_types = [bool, int, float, str, dict]
    items = {}
    for k, v in d.items():
        if type(v) in valid_types:
            if type(v) == dict:
                v = to_sanitized_dict(v)
            items.update({k: v})
        else:
            str(v)
    return items


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






def change_state_dict(state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        if 'attention.output.LayerNorm' in key:
            new_key = key.replace('attention.output.LayerNorm', 'PreAttentionLayerNorm')
            new_keys.append(new_key)
            old_keys.append(key)
        elif 'intermediate' in key:
            new_key = key.replace('dense', 'dense_act.dense')
            new_keys.append(new_key)
            old_keys.append(key)
        elif 'output.LayerNorm' in key:
            new_key = key.replace('output.LayerNorm', 'PostAttentionLayerNorm')
            new_keys.append(new_key)
            old_keys.append(key)
        elif 'pooler' in key:
            new_key = key.replace('dense', 'dense_act.dense')
            new_keys.append(new_key)
            old_keys.append(key)
        elif 'transform' in key:
            new_key = key.replace('dense', 'dense_act.dense')
            new_keys.append(new_key)
            old_keys.append(key)
        # elif 'bert.bert' not in key and 'bert' in key:
        #     new_key = key[5:]
        #     new_keys.append(new_key)
        #     old_keys.append(key)
            
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict

def budget_to_huggingface(state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        if 'embeddings.LayerNorm' in key:
            if 'weight' in key:
                new_key = key.replace('weight','gamma')
                old_keys.append(key)
                new_keys.append(new_key)
            if 'bias' in key:
                new_key = key.replace( 'bias', 'beta')
                old_keys.append(key)
                new_keys.append(new_key)
        if 'PreAttentionLayerNorm' in key:
            if 'weight' in key:
                new_key = key.replace( 'PreAttentionLayerNorm.weight','attention.output.LayerNorm.gamma')
                old_keys.append(key)
                new_keys.append(new_key)
            if 'bias' in key:
                new_key = key.replace( 'PreAttentionLayerNorm.bias','attention.output.LayerNorm.beta')
                old_keys.append(key)
                new_keys.append(new_key)
        if 'PostAttentionLayerNorm' in key:
            if 'weight' in key:
                new_key = key.replace( 'PostAttentionLayerNorm.weight','output.LayerNorm.gamma')
                old_keys.append(key)
                new_keys.append(new_key)
            if 'bias' in key:
                new_key = key.replace( 'PreAttentionLayerNorm.bias','output.LayerNorm.beta')
                old_keys.append(key)
                new_keys.append(new_key)
        if 'intermediate.dense_act' in key:
            new_key = key.replace( 'intermediate.dense_act','intermediate')
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict
    
            