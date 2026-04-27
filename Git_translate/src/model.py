import torch
import torch.nn as nn
from config import *

# 自定义位置编码层
class PositionEncoding(nn.Module):
    def __init__(self, max_len,d_model):
        super.__init__()


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
        self.linear = nn.Linear(in_features=DIM_MODEL,out_features=en_padding_idx)
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
        # 3.Transformer 编码器前向传播
        memory = self.transformer.encoder(src=input,src_key_padding_mask=src_pad_mask)
        # memory 形状(N, S, E)
        return memory
    # 解码方法
    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        return None


if __name__ == '__main__':
    model = TranslationModel(cn_vocab_size=1000,en_vocab_size=1024,cn_padding_idx=0,en_padding_idx=0)
    print(model.encoder)
    print(model.decoder)