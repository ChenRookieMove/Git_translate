# Project Optimization Log

## Project Overview

## Optimization Principles

## Optimization Records

### Optimization 1: 
### Optimization N: <pad>逻辑更改

#### Background
在tokenizer.py模块中进行填充，原来我直接在分词器中制定了一个SEQ_LEN并对所有样本直接<pad>或截断。
```python
    def encode(self,text,seq_len):
        tokens = self.tokenize(text)
        #填充或截断到指定长度(训练时的batch，填充到同一长度才能形成张量，批量计算)
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens)<seq_len:
            tokens = tokens + ([self.pad_token] * (seq_len - len(tokens)))

        ids = [self.word2id.get(token,self.unk_id) for token in tokens]
        return ids
```
#### Problem
（存在什么问题）
样本中出现了大量的<pad>。

#### Analysis
（为什么会有这个问题）
SEQ_LEN取值是按照样本中较长的句子长度来取的。
并且对于每一个句子都按照这一个长度来填充或截断
#### Solution
（怎么改）
将填充这一步放在dataloader里来做，对于取的每一个batch都动态的取一个SEQ_LEN来进行填充。


#### Implementation
（具体代码改在哪）
```python
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # 将inputs和targets分成两个列表
    # batch形状[(input0,target0),(input1,target1),(input2,target2)....]
    input_tensor_list = [item[0] for item in batch]
    target_tensor_list = [item[1] for item in batch]
    # 合并成长度对齐的一个batch tensor
    input_batch_tensor = pad_sequence(input_tensor_list,batch_first=True,padding_value=0 )
    target_batch_tensor = pad_sequence(target_tensor_list,batch_first=True,padding_value=0 )
    return input_batch_tensor,target_batch_tensor
```
#### Result
（优化效果）

#### Trade-off
（代价）











### Optimization 2: xxx

### Optimization 3: xxx

## Performance Comparison

## Future Optimization Directions