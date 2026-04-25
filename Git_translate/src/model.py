import torch
import torch.nn as nn
from config import *




## 自定义编码器(基于GRU)
class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx = padding_idx)
        # 单层单向GRU
        self.gru =nn.GRU(EMBEDDING_SIZE,HIDDEN_SIZE,batch_first=True)

    def forward(self,x):
        # 词嵌入
        embed = self.embedding(x)
        #GRU前向传播，得到隐状态(N，L，hidden_size)
        output,_ = self.gru(embed)
        #取真实的最后一个隐状态，作为上下文的特征向量
        indices = torch.arange(output.shape[0])
        lengths = (x!= self.embedding.padding_idx).sum(dim=1)  #计算每个序列的真实长度
        # 重点：用列表索引，来取出真实的最后一个隐状态,降维了，得到[N,hidden_size]
        features = output[indices,lengths-1]
        return features
#自定义解码器
class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True)
        #全连接层：
        self.linear = nn.Linear(in_features=HIDDEN_SIZE,out_features=vocab_size)
    #前向传播,传入初始隐状态
    def forward(self, x,h0=None):
        # 1.词嵌入
        embed = self.embedding(x)
        # 2. x与h0在GRU前向传播，得到输出隐状态(N，L，hidden_size)--没有填充且为单层，所以输出的隐状态就是特征向量
        # gru输入格式(batch, seq_len, embedding_dim)
        output, hn = self.gru(embed,h0)
        # 3. 整合特征，预测分类,这里只需保证最后最后一维能和linear对上就行，如果展开对应一定是又深意的。
        output = self.linear(output)
        return output,hn
#自定义Seq2Seq模型
class TranslationModel(nn.Module):
    #初始化
    def __init__(self,cn_vocab_size,en_vocab_size,cn_padding_idx,en_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(cn_vocab_size,cn_padding_idx)
        self.decoder = TranslationDecoder(en_vocab_size,en_padding_idx)

if __name__ == '__main__':
    model = TranslationModel(cn_vocab_size=1000,en_vocab_size=1024,cn_padding_idx=0,en_padding_idx=0)
    print(model.encoder)
    print(model.decoder)

