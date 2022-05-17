from typing import DefaultDict
import sys
import torch
import os
import tqdm
from collections import Counter
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def get_layer_num(filename):
    model = torch.load(filename, map_location='cpu')['module']
    enc_keys = [x for x in model.keys() if 'ff.dense_relu_dense.wi.weight' in x and 'encoder' in x]
    dec_keys = [x for x in model.keys() if 'ff.dense_relu_dense.wi.weight' in x and 'decoder' in x]

    enc_nums = [int(x.split('.')[2]) for x in enc_keys]
    dec_nums = [int(x.split('.')[2]) for x in dec_keys]

    return max(enc_nums)+1, max(dec_nums)+1

def load_ffn_weight(filename, template, layer):

    model = torch.load(filename, map_location='cpu')
    key = template.format(layer)

    return model[key].numpy()

def load_hidden_states(folder, layer):
    sub_folder = 'hiddens'

    target = os.path.join(folder, "{}_layer_{}".format(sub_folder, layer))

    vecs = []
    if os.path.exists(target):
        vecs = torch.load(target)
    else:
        files = os.listdir(os.path.join(folder,sub_folder))
        files = sorted(files, key=lambda x: int(x))
        print(files)
        if 'race' in target:
            files = files[:len(files)//10]
        if 'squad' in target:
            files = files[:len(files)//5]
        for filename in tqdm.tqdm(files):
            path = os.path.join(folder,sub_folder, filename)
            hiddens = torch.load(path, map_location='cpu')
            vecs.append(hiddens[layer])
        torch.save(vecs, target)
    return vecs

class ModelConfig:

    def __init__(self, filename, folder, split_num):
        self.filename = filename
        self.folder = folder
        self.split_num = split_num

class LayerSplit:

    def __init__(self, config : ModelConfig, template, layer=0):
        self.config = config
        self.layer = layer
        self.template = template

    def split(self):
        pass
    
    def save(self):
        save_folder = os.path.join(self.config.folder, self.type)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        filename = os.path.join(save_folder, self.template.format(self.layer))
        torch.save(self.labels, filename)

    def cnt(self):
        print(Counter(self.labels))

    def load_param(self):
        self.ffn_weight = load_ffn_weight(self.config.filename, self.template, self.layer)
        self.neuron_num = self.ffn_weight.shape[0]
        self.split_size = self.neuron_num // self.config.split_num
        assert self.split_size * self.config.split_num == self.neuron_num

class RandomSplit(LayerSplit):
    
    def __init__(self, config: ModelConfig, layer=0, is_encoder=True):
        super().__init__(config, layer=layer, is_encoder=is_encoder)
        self.type = 'random_split'

    def split(self):
        self.load_param()

        self.labels = [i // self.split_size for i in range(self.neuron_num)]

class ParamSplit(LayerSplit):
    
    def __init__(self, config: ModelConfig, template, layer=0):
        super().__init__(config, template=template, layer=layer)
        self.type = 'param_split'

    def split(self):
        from clustering.equal_groups import EqualGroupsKMeans
        self.load_param()
        ffn_weight_norm = sklearn.preprocessing.normalize(self.ffn_weight)
        
        kmeans = EqualGroupsKMeans(n_clusters=self.config.split_num, n_jobs=-1, n_init=1, max_iter=20, verbose=1).fit(ffn_weight_norm, None)
        
        self.labels = [x for x in kmeans.labels_]

class BlockCenter:

    def __init__(self, config, template, filename):
        self.config = config
        self.filename = filename
        self.labels = torch.load(filename)
        self.template = template
        
        basename = os.path.basename(filename)
        vecs = basename.split('_')
        self.layer = int(vecs[-1])

    def cal_center(self):
        pass

    def save(self):
        print(self.centers.shape)
        torch.save(self.centers, "{}_{}".format(self.filename, self.type))
        self.save_acc()

    def save_acc(self):
        with open("{}_{}_acc".format(self.filename, self.type), 'w') as fout:
            fout.write(str(self.acc))

class RandomCenter(BlockCenter):

    def __init__(self, config, filename):
        super().__init__(config, filename)
        self.type = "random"
    
    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.layer, self.is_encoder)
        ffn_weight_norm = ffn_weight

        d = {}
        for i, x in enumerate(self.labels):
            if x not in d:
                d[x] = ffn_weight_norm[i, :]
        centers = sorted(list(d.items()), key=lambda x: x[0])
        
        self.centers = sklearn.preprocessing.normalize(np.array([x[1] for x in centers]))
        self.acc = 0
    
class ParamCenter(BlockCenter):

    def __init__(self, config, filename):
        super().__init__(config, filename)
        self.type = "param"
    
    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.layer, self.is_encoder)
        ffn_weight_norm = sklearn.preprocessing.normalize(ffn_weight)

        centers = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            centers.append(ffn_weight_norm[np.array(self.labels) == i, :].mean(0))

        centers = np.array(centers)
        self.centers = centers

        centers = torch.tensor(centers).cuda().unsqueeze(0)

        patterns = []
        for i in range(num_blocks):
            patterns.append(np.array(self.labels) == i)
        patterns = torch.Tensor(patterns).cuda().float().transpose(0, 1) # 4096, num_blocks

        acc = []
        hiddens = load_hidden_states(self.config.folder, self.layer, self.is_encoder)
        hiddens = torch.cat(hiddens, 0).float()
        hiddens = hiddens.view(-1, hiddens.shape[-1])
        hiddens = hiddens / torch.norm(hiddens, dim=-1).unsqueeze(-1)
        num = hiddens.shape[0]

        ffn_weight = torch.tensor(ffn_weight).cuda().transpose(0, 1).float()
        for i in range(num // 10 * 9, num, 512):
            with torch.no_grad():
                input = hiddens[i:i+512, :].cuda()
                acts = torch.relu((torch.matmul(input, ffn_weight))) # 512, 4096
                scores = torch.matmul(acts, patterns) # 512, num_blocks, vary from 0 to 1
                labels = torch.topk(scores, k=25, dim=-1)[1]

                input = input / torch.norm(input, dim=-1).unsqueeze(-1)
                dist = -1 * torch.norm(input.unsqueeze(1).expand(-1, num_blocks, -1) - centers, dim=-1)
                pred = torch.topk(dist, k=25, dim=-1)[1]

                for x, y in zip(labels, pred):
                    x = set(x.cpu().numpy())
                    y = set(y.cpu().numpy())
                    acc.append(len(x & y) / 25)
        print("param acc", np.mean(acc))
        sys.stdout.flush()
        self.acc = np.mean(acc)

class MLPCenter(BlockCenter):
    def __init__(self, config, template, filename):
        super().__init__(config, template, filename)
        self.type = "input_compl"
    
    def cal_center(self):
        ffn_weight = load_ffn_weight(self.config.filename, self.template, self.layer)
        ffn_weight_norm_ = sklearn.preprocessing.normalize(ffn_weight)
        centers = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            centers.append(ffn_weight_norm_[np.array(self.labels) == i, :].mean(0))
        centers = np.array(centers) # num_blocks, 1024

        ffn_weight = torch.tensor(ffn_weight).cuda().transpose(0, 1).float()
        patterns = []
        num_blocks = max(self.labels) + 1
        for i in range(num_blocks):
            patterns.append(np.array(self.labels) == i)
        patterns = torch.Tensor(patterns).cuda().float().transpose(0, 1)

        hiddens = load_hidden_states(self.config.folder, self.layer)
        hiddens = torch.cat(hiddens, 0)
        hideen_size = hiddens.shape[1]
        hiddens = hiddens.transpose(1, 2).reshape(-1, hideen_size)

        hiddens = hiddens / torch.norm(hiddens, dim=-1).unsqueeze(-1)

        model = torch.nn.Sequential(torch.nn.Linear(hiddens.shape[-1], num_blocks, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(num_blocks, num_blocks, bias=False))

        def weights_init(m):
            if isinstance(m, torch.nn.Linear):
                if m.weight.shape[-1] == hiddens.shape[-1]:
                    m.weight.data = torch.from_numpy(centers).float()
                else:
                    m.weight.data = torch.eye(m.weight.data.shape[0])
                    #torch.nn.init.normal_(m.weight.data)
                #m.bias.data[:] = 0

        model.apply(weights_init)
        
        model.cuda()

        #for name, param in model.named_parameters():
        #    if param.shape[-1] == hiddens.shape[-1]:
        #        param.requires_grad = False

        optim = torch.optim.Adam(model.parameters(), lr=0.01)

        loss_func = torch.nn.BCEWithLogitsLoss()

        save_acc = [0, 0]
        save_epoch = [-1, -1]

        self.centers = model

        train_hiddens = hiddens[:hiddens.shape[0] // 10 * 9, :]
        #pos_max = None

        last_epoch = -1

        for epoch in range(30):
            train_hiddens=train_hiddens[torch.randperm(train_hiddens.size()[0])]

            pbar = tqdm.tqdm(range(0, train_hiddens.shape[0], 512))
            for i in pbar:
                model.zero_grad()

                input = train_hiddens[i:i+512, :].float().cuda()
                with torch.no_grad():
                    acts = torch.relu((torch.matmul(input, ffn_weight))).float() # 512, 4096
                    scores = torch.matmul(acts, patterns)
                    #if pos_max is None:
                    #    pos_max = torch.max(scores).item()
                    #else:
                    #    pos_max = max([torch.max(scores).item(), pos_max])
                    #scores /= pos_max # 512, num_blocks, vary from 0 to 1
                    scores /= scores.max()
                pred = model(input)
                loss = loss_func(pred.view(-1), scores.view(-1))
              
                loss.backward()
                optim.step()

                pbar.set_description("loss: {:.4f}".format(loss.item()))

            acc = []
            
            # for i in range(hiddens.shape[0] // 10 * 9, hiddens.shape[0], 512):
            # for i in range(0, hiddens.shape[0], 512):
            for i in range(0, 512, 512):
                with torch.no_grad():
                    input = hiddens[i:i+512, :].float().cuda()
                    acts = torch.relu((torch.matmul(input, ffn_weight))).float() # 512, 4096

                    scores = torch.matmul(acts, patterns) # 512, num_blocks, vary from 0 to 1
                    mask, labels = torch.topk(scores, k=int(num_blocks*0.2), dim=-1)
                    mask = mask > 0
                    
                    pred = model(input)
                    pred = torch.topk(pred, k=int(num_blocks*0.2), dim=-1)[1]

                    for x, m, s in zip(labels, mask, scores):
                        if m.sum().item() == 0:
                            continue
                        x = sum([s[xx] for xx in x.cpu()]).item()
                        y = s.sum().item()
                        acc.append( x / y)
            
            cur_acc = np.mean(acc)
            if cur_acc > save_acc[0]:
                self.del_ckpt(save_epoch[1])
                save_acc = [cur_acc, save_acc[0]]
                save_epoch = [epoch, save_epoch[0]]
                print("input compl center acc", np.mean(acc))
                self.acc = save_acc[1]
                sys.stdout.flush()
                self.save(epoch)
            elif cur_acc > save_acc[1]:
                self.del_ckpt(save_epoch[1])
                save_acc = [save_acc[0], cur_acc]
                save_epoch = [save_epoch[0], epoch]
                print("input compl center acc", np.mean(acc))
                self.acc = save_acc[1]
                sys.stdout.flush()
                self.save(epoch)
        os.system("rm -rf {}_{}_{}".format(self.filename, self.type, save_epoch[0]))
        os.system("cp {0}_{1}_{2} {0}_{1}".format(self.filename, self.type, save_epoch[1]))
        os.system("rm {0}_{1}_{2}".format(self.filename, self.type, save_epoch[1]))
    
    def del_ckpt(self, epoch):
        os.system("rm -rf {}_{}_{}".format(self.filename, self.type, epoch))

    def save(self, epoch):
        print("input compl center save")
        torch.save(self.centers, "{}_{}_{}".format(self.filename, self.type, epoch))
        self.save_acc()
