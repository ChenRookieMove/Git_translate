import torch
import torch.nn as nn
from config import *
import math

# 自定义位置编码层
class PositionEncoding(nn.Module):
    # 初始化，生成位置编码矩阵(L,E)
    def __init__(self, max_len,d_model):
        super().__init__()
        # 初始化编码矩阵
        pe = torch.zeros(max_len,d_model,dtype=torch.float)
        # 遍历每一行 (每个位置pos)
        for pos in range(max_len):
            # 遍历当前位置向量的每个特征，步长为 2
            for _2i in range(0,d_model,2):
                # 按公式计算向量里的这两个特征
                pe[pos, _2i] = math.sin(pos/(10000**(_2i/d_model)))
                pe[pos, _2i+1] = math.cos(pos/(10000**(_2i/d_model)))
        self.register_buffer('pe',pe)
    def forward(self,x):
        # 这里对于一个batch L的长度可能不是max_len(SEQ_LEN),注意区分，transformer里的宽度一致指的是词向量
        # x 形状：(N, L, E =d_model)
        seq_len = x.shape[1] # 提取当先序列长度L
        # 在位置编码矩阵中截取 L 个向量
        part_pe = self.pe[0:seq_len]
        # pe形状：(L,E)，广播pe
        return x +  part_pe



#自定义Seq2Seq模型
class TranslationModel(nn.Module):
    #初始化
    def __init__(self,cn_vocab_size,en_vocab_size,cn_padding_idx,en_padding_idx):
        super().__init__()
        # 词嵌入层
        self.cn_embedding = nn.Embedding(cn_vocab_size,embedding_dim=DIM_MODEL,padding_idx=cn_padding_idx)
        self.en_embedding = nn.Embedding(en_vocab_size,embedding_dim=DIM_MODEL,padding_idx=en_padding_idx)
        # 位置编码
        self.position_encoding = PositionEncoding(SEQ_LEN,DIM_MODEL)
        # Transformer层
        self.transformer = nn.Transformer(
            d_model = DIM_MODEL,
            nhead = NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            batch_first=True,
        )
        # 输出线性层
        self.linear = nn.Linear(in_features=DIM_MODEL,out_features=en_vocab_size)
    # 前向传播,将transformer 需要的参数全部传入
    def forward(self, src, tgt, src_pad_mask, tgt_mask):
        # 输入源序列（N,S）,目标序列(N,T)
        # 编码
        memory = self.encode(src,src_pad_mask)
        # 解码
        output = self.decode(tgt,memory,tgt_mask,memory_pad_mask=src_pad_mask)

        return output
    # 编码方法
    def encode(self, src, src_pad_mask):
        # src形状: (N,S),src_pad_mask 形状：(N,S)
        # 1.词嵌入
        embed = self.cn_embedding(src)
        # embed 形状：(N,S,E=d_model)
        # 2. 叠加位置编码
        input = self.position_encoding(embed)
        # input 形状：(N, S, E)
        # 3.Transformer解码器前向传播
        memory = self.transformer.encoder(src=input,src_key_padding_mask=src_pad_mask)
        # memory 形状(N, S, E)
        return memory
    # 解码方法
    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        # tgt形状: (N,T=tgt_len ),tgt_mask 形状：(T,T)
        # 1.词嵌入
        embed = self.en_embedding(tgt)
        # embed 形状：(N,T,E=d_model)
        # 2. 叠加位置编码
        input = self.position_encoding(embed)
        # input 形状：(N, T, E)
        # 3.Transformer 编码器前向传播
        output = self.transformer.decoder(
            tgt=input,
            memory = memory,
            tgt_mask = tgt_mask,
            memory_key_padding_mask = memory_pad_mask,
        )
        # output 形状(N, T, E)
        # 4. 经过输出线性层整合，得到预测输出
        output = self.linear(output)
        # output 形状：(N,T,en_vocab_size)
        return output


if __name__ == '__main__':
    model = TranslationModel(cn_vocab_size=1000,en_vocab_size=1024,cn_padding_idx=0,en_padding_idx=0)
    print(model)
