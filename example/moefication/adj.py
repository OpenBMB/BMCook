import utils
import numpy as np
import torch
import tqdm
import sys
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='results/t5-base/ckpt.bin', help='path to the model checkpoint')
parser.add_argument('--res_path', type=str, default='results/t5-base/', help='path to store the results of moefication')
parser.add_argument('--num-layer', type=int, default=12, help='number of layers')
parser.add_argument('--templates', type=str, default='encoder.blocks.{}.ff.dense_relu_dense.wi.weight,decoder.blocks.{}.ff.dense_relu_dense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')
parser.add_argument('--num-gpu', type=int, default=4, help='number of gpus')

args = parser.parse_args()

num_layer = args.num_layer
batch_size = 8
max_instance = 200000

def run(proc_id):
    proc_id, model_path, res_path, template, num_gpu = proc_id
    cuda_dev = torch.device('cuda:{}'.format(proc_id))
    for layer in range(num_layer):
        layer_id = + layer
        if layer_id % num_gpu != proc_id:
            continue

        ffn = torch.tensor(utils.load_ffn_weight(model_path, template, layer))
        hidden = utils.load_hidden_states(res_path, layer)
        hidden = torch.cat(hidden, 0).transpose(1, 2).reshape(-1, 4096)

        cnt = 0
        adj = torch.zeros(ffn.shape[0], ffn.shape[0], device=cuda_dev).float()
        ffn = torch.tensor(ffn).to(cuda_dev).transpose(0, 1)
        for i in tqdm.tqdm(range(hidden.shape[0]//batch_size)):
            with torch.no_grad():
                dat = hidden[i*batch_size:(i+1)*batch_size].to(cuda_dev) 
                res = torch.nn.functional.relu(torch.matmul(dat, ffn)).unsqueeze(-1)
                res = torch.clamp(torch.bmm(res, res.transpose(1, 2)).sum(0), max=1)
                adj += res
        
            cnt += batch_size
            if cnt > max_instance:
                break
        del hidden

        adj = adj.cpu().numpy()
        target = os.path.join(res_path, template.format(layer))

        threshold = 0
        pos = 10
        while threshold == 0:
            assert pos != 110
            threshold = np.percentile(adj.reshape(-1), pos)
            pos += 10
        print("threshold", threshold, layer_id, pos, adj.max())
        threshold = threshold * 0.99
        adj /= threshold

        with open(target, "w") as fout:
            edges = 0
            for i in range(adj.shape[0]):
                cnt = 0
                for j in range(adj.shape[1]):
                    if i == j or adj[i, j] < 1:
                        pass
                    else:
                        cnt += 1
                edges += cnt
            assert edges > 0
            fout.write("{} {} {}\n".format(adj.shape[0], edges // 2, "001"))
            for i in range(adj.shape[0]):
                vec = []
                for j in range(adj.shape[1]):
                    if i == j or adj[i, j] < 1:
                        pass
                    else:
                        val = int(adj[i, j])
                        vec.append([j+1, val])
                fout.write(" ".join(["{} {}".format(x[0], x[1]) for x in vec]) + "\n")

import multiprocessing
templates = args.templates.split(',')

for template in templates:
    pool = multiprocessing.Pool(processes=args.num_gpu)
    pool.map(run, [(i, args.model_path, args.res_path, template, args.num_gpu) for i in range(args.num_gpu)])
    pool.close()
    pool.join()

