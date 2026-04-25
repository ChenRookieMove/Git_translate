import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from config import *
from torch.nn.utils.rnn import pad_sequence # rnn中专门做序列填充的方法

# 数据集类，继承dataset类。
class TranslateDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')
    def __len__(self):
        return len(self.data)
    # 从数据集中，拿到每一个条对应的输入中文，和输出的英文
    def __getitem__(self,index):
        input = torch.tensor(self.data[index]['cn'],dtype=torch.long)
        target = torch.tensor(self.data[index]['en'],dtype=torch.long)
        return input,target
# 定义一个整理函数，将一批函数长度对齐（填充）
def collate_fn(batch):
    # 将inputs和targets分成两个列表
    # batch形状[(input0,target0),(input1,target1),(input2,target2)....]
    input_tensor_list = [item[0] for item in batch]
    target_tensor_list = [item[1] for item in batch]
    # 合并成长度对齐的一个batch tensor
    input_batch_tensor = pad_sequence(input_tensor_list,batch_first=True,padding_value=0 )
    target_batch_tensor = pad_sequence(target_tensor_list,batch_first=True,padding_value=0 )
    return input_batch_tensor,target_batch_tensor

def get_dataloader(train=True):
    path =PROCESSED_DATA_DIR/(TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    dataset = TranslateDataset(path)
    # 在dataloader里统一同一batch的size
    dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    return dataloader
if __name__ == '__main__':
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    for input , target in train_dataloader:
        print(input.shape,target.shape)
        break