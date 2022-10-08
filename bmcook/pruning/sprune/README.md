## note

怎样解决跨粒度sparsity计算问题?

1. 可以用一个字典存储所有的待训练mask及param信息，该列表索引是overall index，记录一个score和对应的param。
2. 遍历所有需训练的mask，放在对应的overall index上，如果在该overall index上重复出现则score累乘，关于param，永远保留最底层的param
3. 所以最后的得到的是:[{param: int, score: float} * overall_layers]
4. 之所以选用overall_index而不用transformer_index是因为所有粒度模块的overall_index是统一的

怎样解决actual prune的问题？

1. 首先加载plugin和state_dict
2. 从transformer开始，以树状向下延伸，直到叶子节点