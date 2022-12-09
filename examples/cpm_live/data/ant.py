import os
import torch
import random
import struct
import numpy as np
import bmtrain as bmt
import torch.utils.data as data
from itertools import accumulate

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.double,
    8: np.uint16
}

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

def index_file_path(prefix_path):
    return prefix_path + '.idx'

def data_file_path(prefix_path):
    return prefix_path + '.bin'

class DistributedMMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b'MMIDIDX\x00\x00'
        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, name, rank_number, rank_total, cache = None):
        
        super().__init__()

        self._path = path
        self._name = name
        self._state = 0
        if cache is not None:
            self._cache = cache
            os.makedirs(self._cache, exist_ok=True)
        else:
            self._cache = None
        self._rank_total = rank_total
        self._rank_number = rank_number
        self._index = None
        self._bin_buffer = None
        self._bin_buffer_mmap = None
        self.history = {-1:0}

        self._do_init(self._path, self._name, self._cache, self._state)

    def __getstate__(self):
        return self._path + self._name + "_%d"%(self._state)

    def __setstate__(self, state):
        self._state = state
        self._do_init(self._path, self._name, self._cache, self._state)

    def _do_init(self, path, name, cache, state):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

        self._state = state

        source_file = path + name + "_%d"%(self._state)

        self._index = self.Index(index_file_path(source_file))
        self._bin_buffer_mmap = np.memmap(data_file_path(source_file), mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        self.history[state] = self.history[state - 1] + len(self._index) // self._rank_total
        self.start = (len(self._index) // self._rank_total) * self._rank_number
        print (self.history[state],"======================", self.start, self.history[self._state - 1])

    def __del__(self):
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap
        if self._index is not None:
            del self._index

    def _next_file(self):
        self._state += 1
        bmt.print_rank(f"next_file: {self._state}")
        self._do_init(self._path, self._name, self._cache, self._state)
    
    def __relative_idx(self, idx):
        res = self.start + idx - self.history[self._state - 1]
        return res

    def __slice_item(self, start, stop):
        ptr = self._index._pointers[self.__relative_idx(start)]
        sizes = self._index._sizes[self.__relative_idx(start):self.__relative_idx(stop)]
        offsets = list(accumulate(sizes))
        np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=sum(sizes), offset=ptr)
        return np.split(np_array, offsets[:-1])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            while idx >= self.history[self._state]:
                self._next_file()
            ptr, size = self._index[self.__relative_idx(idx)]
            return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(2147483647)
            assert step == 1 or step is None, "Slices into indexed_dataset must be contiguous"
            if stop >= self.history[self._state]:
                res_1 = self.__slice_item(start, self.history[self._state]) 
                self._next_file()
                res_2 = self.__slice_item(self.history[self._state - 1], stop)
                return res_1 + res_2
            else:
                return self.__slice_item(start, stop)

    @property
    def sizes(self):
        return self._index.sizes
        
    def exists(path):
        return (os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path)))

class CPMAnt_Dataset(data.Dataset):
    def __init__(self, ctx, max_length = 1024, prompt_length = 32, tokenizer = None):
        self.ctx = ctx
        self.max_length = max_length + prompt_length
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ctx)

    def __get_item_data(self, raw_data, index):

        global_task = raw_data[0]
        n_segment = raw_data[1]
        len_info = n_segment * 3 + 2
        segment_len  = raw_data[2:len_info:3] 
        segment_type = raw_data[3:len_info:3]
        segment_task = raw_data[4:len_info:3]
        ctx = raw_data[len_info:]

        if ctx.shape[0] > self.max_length - self.prompt_length:
            return None, None, None, None, None, None, None
        len_ctx = min(ctx.shape[0], self.max_length - self.prompt_length)

        context_inp = np.full(len_ctx, True)
        position_inp = np.arange(len_ctx, dtype=np.int64)
        segment_inp = np.full(len_ctx, 0, dtype=np.int64)
        task_inp = np.full(len_ctx, 0, dtype=np.int64)
        tgt = np.full(len_ctx, -100, dtype=np.int64)

        # for each segment
        segment_begin = 0
        for i in range(n_segment):
            segment_end = segment_begin + segment_len[i]
            task = segment_task[i]
            # generate target
            if task == 0:
                num_mask = random.randint(1, segment_len[i] - 1)
                mask_idx = np.random.choice(segment_len[i] - 1, num_mask, replace=False) + segment_begin
                context_inp[mask_idx + 1] = False
                task_inp[segment_begin:segment_end] = task
                assert segment_type[i] == 1
            elif task == 1:
                num_mask = random.randint(1, segment_len[i] - 1)
                context_inp[segment_end-num_mask:segment_end] = False
                task_inp[segment_begin:segment_end] = task
                assert segment_type[i] == 2
            segment_inp[segment_begin:segment_end] = segment_type[i]
            tgt[segment_begin : segment_end - 1] = np.where(
                context_inp[segment_begin + 1 : segment_end],
                -100,
                ctx[segment_begin + 1 : segment_end]
            )
            segment_begin = segment_end
        # prepend prompt segment
        context_inp = np.concatenate((np.full(self.prompt_length, True), context_inp))
        position_inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64), position_inp + self.prompt_length))
        segment_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), segment_inp))
        task_inp = np.concatenate((np.full(self.prompt_length, 0, dtype=np.int64), task_inp))
        tgt = np.concatenate((np.full(self.prompt_length, -100, dtype=np.int64), tgt))
        inp = np.concatenate((np.arange(self.prompt_length, dtype=np.int64) + self.prompt_length * global_task, ctx))
        return inp, tgt, inp.shape[0], context_inp, position_inp, segment_inp, task_inp

    def __getitem__(self, index):
        ctx = self.ctx[index]
        th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx = \
                self.__get_item_data(ctx, index)
        return th_ctx, th_tgt, len_ctx, context_ctx, position_ctx, segment_ctx, task_ctx

class BatchPacker:
    def __init__(self, dataset, max_length, batch_size):
        self.last_idx = 0
        self.dataset = dataset
        self.max_length = max_length
        self.batch_size = batch_size
    
    def state(self):
        return self.last_idx
    
    def load_state(self, state):
        self.last_idx = state

    def __iter__(self):
        st = self.last_idx
        ctx = []
        tgt = []
        context = []
        position = []
        segment = []
        span = []
        task_info = []

        while True:
            ctx_data, tgt_data, _len, context_data, position_data, segment_data, task_data = self.dataset[st]
            st += 1
            if ctx_data is None:
                continue
            assert _len <= self.max_length

            ctx_data = ctx_data.astype("int64")
            tgt_data = tgt_data.astype("int64")

            for index in range(len(ctx)):
                if span[index][-1] + _len < self.max_length:
                    ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                    tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                    context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                    position[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(position_data)[:_len].long()
                    segment[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(segment_data)[:_len].long()
                    task_info[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(task_data)[:_len].long()
                    span[index].append(span[index][-1] + _len)
                    break
            else:
                _ctx = torch.zeros((self.max_length,), dtype=torch.long)
                _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
                _tgt = torch.full((self.max_length,), -100, dtype=torch.long)
                _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
                _context = torch.full((self.max_length,), False, dtype=torch.bool)
                _context[:_len] = torch.from_numpy(context_data)[:_len].bool()
                _position = torch.full((self.max_length,), False, dtype=torch.long)
                _position[:_len] = torch.from_numpy(position_data)[:_len].long()
                _segment = torch.full((self.max_length,), False, dtype=torch.long)
                _segment[:_len] = torch.from_numpy(segment_data)[:_len].long()
                _task_info = torch.full((self.max_length,), -1, dtype=torch.long)
                _task_info[:_len] = torch.from_numpy(task_data)[:_len].long()
                ctx.append(_ctx)
                tgt.append(_tgt)
                context.append(_context)
                position.append(_position)
                segment.append(_segment)
                task_info.append(_task_info)
                span.append([_len])

            if len(ctx) > self.batch_size:
                _span = torch.zeros((self.batch_size, self.max_length + 1), dtype=torch.long)
                for bindex in range(self.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1
                
                self.last_idx = st
                yield {
                    "ctx": torch.stack(ctx[:self.batch_size]),
                    "tgt": torch.stack(tgt[:self.batch_size]),
                    "context": torch.stack(context[:self.batch_size]),
                    "segment": torch.stack(segment[:self.batch_size]),
                    "position": torch.stack(position[:self.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:self.batch_size]]),
                    "task": torch.stack(task_info[:self.batch_size]),
                }

                ctx = ctx[self.batch_size:]
                tgt = tgt[self.batch_size:]
                context = context[self.batch_size:]
                segment = segment[self.batch_size:]
                position = position[self.batch_size:]
                span = span[self.batch_size:]
                task_info = task_info[self.batch_size:]

