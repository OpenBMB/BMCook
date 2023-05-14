"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import time
import torch

from data import indexed_dataset
from transformers import GPT2Tokenizer

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    def encode(self, line):
        data = line
        ids = {}
        doc_ids = Encoder.tokenizer(data)['input_ids']
        ids['text'] = [doc_ids]
        return ids, len(line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input file')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--max-num', type=int, default=0)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    # def pack_doc(fin):
    #     doc = ""
    #     for line in fin:
    #         if line == "\n":
    #             if len(doc) > 0:
    #                 yield doc
    #             doc = ""
    #         else:
    #             doc += " " + line.strip()

    encoded_docs = pool.imap(encoder.encode, fin, 25)

    level = "document"

    print(f"Output prefix: {args.output_prefix}")
    output_bin_file = "{}_{}.bin".format(args.output_prefix, level)
    output_idx_file = "{}_{}.idx".format(args.output_prefix, level)
    builder = indexed_dataset.make_builder(output_bin_file,
                                            impl=args.dataset_impl,
                                            vocab_size=len(tokenizer))

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        builder.add_item(torch.IntTensor(doc['text']))
        builder.end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
        if args.max_num > 0 and i == args.max_num:
            break

    builder.finalize(output_idx_file)

if __name__ == '__main__':
    main()
