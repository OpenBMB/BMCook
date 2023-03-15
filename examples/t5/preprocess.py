# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for pretraining."""

import argparse
import io
import json
import multiprocessing
import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from transformers import T5Tokenizer

from data import indexed_dataset

# from ray.util.multiprocessing.pool import Pool

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        Encoder.splitter = IdentitySplitter()

    def encode(self, line):
        # end with <eod>
        if len(line) > 5000000:
            return None, None, None, 0
        data = line.strip()
        data = data.replace("<n>", "\n")

        contexts = []
        targets = []

        # split data into multiple texts
        words = data.split(" ")
        while len(words) > self.args.max_seq_length:

            doc_ids = Encoder.tokenizer.encode(" ".join(words[:self.args.max_seq_length]), max_length=self.args.max_seq_length-1, add_special_tokens=False)

            words = words[self.args.max_seq_length:]

            if len(doc_ids) < 10:
                continue

            span_start_ends = self.random_spans_noise_mask(
                len(doc_ids), noisy_density=self.args.noisy_density, mean_noise_span_length=self.args.mean_noise_span_length)

            start = 0
            context = []
            target = []
            for i, span in enumerate(span_start_ends):
                # "+[0]" placeholder for sentinel
                context.extend(doc_ids[start:span[0]] + [self.tokenizer.vocab_size-i-1])
                target.extend([self.tokenizer.vocab_size-i-1] + doc_ids[span[0]:span[1]])
                start = span[1]
            
            assert start == len(doc_ids)

            target = [0] + target + [self.tokenizer.vocab_size-i-2, 1]
            context += [1]

            print(context, target)
            contexts.append(context)
            targets.append(target)

        return contexts, targets, len(line)

    def random_spans_noise_mask(self, length, noisy_density=0.15, mean_noise_span_length=10.0):
        num_noise_tokens = round(length * noisy_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1, generator=g)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))

            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="openwebtext.txt", type=str, help='Path to input TXT')
    
    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="pretrain_data", type=str)
    group.add_argument('--output_prefix', default="webtext", type=str,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=32,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--noisy_density', type=float, default=0.15)
    group.add_argument('--max_seq_length', type=int, default=512)
    group.add_argument('--mean_noise_span_length', type=int, default=3)

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)

    fin = open(args.input, 'r')
    encoder = Encoder(args)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, fin, 30)

    level = "document"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    context_bin_file = os.path.join(args.output_path, "{}_{}_context.bin".format(args.output_prefix, level))
    context_idx_file = os.path.join(args.output_path,  "{}_{}_context.idx".format(args.output_prefix, level))
    target_bin_file = os.path.join(args.output_path,  "{}_{}_target.bin".format(args.output_prefix, level))
    target_idx_file = os.path.join(args.output_path,  "{}_{}_target.idx".format(args.output_prefix, level))

    # set dtype == int64
    # max idx: 2 ** 63 - 1
    builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl, dtype=np.int32)
    builder_target = indexed_dataset.make_builder(target_bin_file, impl=args.dataset_impl, dtype=np.int32)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    # sentinel_idx = tokenizer.vocab_size # start from the last token of the tokenizer
    print("tokenizer vocab size:", tokenizer.vocab_size)
    for i, (contexts, targets, bytes_processed) in enumerate(encoded_docs, start=1):
        if len(contexts) == 0 or len(targets) == 0:
            continue
        total_bytes_processed += bytes_processed

        for tmp_context, temp_target in zip(contexts, targets):
            builder_context.add_item(torch.IntTensor(tmp_context))
            builder_target.add_item(torch.IntTensor(temp_target))
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder_context.finalize(context_idx_file)
    builder_target.finalize(target_idx_file)

    pool.close()

if __name__ == '__main__':
    main()
