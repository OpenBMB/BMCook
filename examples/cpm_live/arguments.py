# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
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

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", type=str, help="model configuration file")
    p.add_argument("--load", type=str, help="model checkpoint")
    p.add_argument('--teacher-config', type=str, help="teacher configuration file")
    p.add_argument('--load-teacher', type=str, help="model checkpoint")
    p.add_argument('--log-interval', type=int, default=1, help='Log loss, LR, scale every n steps')
    p.add_argument('--save-interval', type=int, default=1000, help='Interval steps between checkpoints')
    p.add_argument('--start-lr', type=float, default=1e-4, help='Start learning rate')
    p.add_argument('--data-path', default='openwebtext_document', help='Path to dataset')
    p.add_argument('--cook-config', type=str, help='Path to BMCook config file')
    
    return p.parse_args()

