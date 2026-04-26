import torch
import torch.nn as nn
from config import *

# 自定义Attention类
class Attention(nn.Module):
    # 前向传播
    def forward(self,decoder_hiddens,encoder_outputs):
        # 1.计算注意力评分
        # 输出形状：decoder_hiddens(N,L_dec,hidden_size),encoder_outputs(N,L_enc,hidden_size)
        attention_scores = torch.bmm(decoder_hiddens,encoder_outputs.transpose(1,2))
        # 评分形状 (N,L_dec,L_enc) --每个时间步都有一个长度为L_enc的注意力评分
        # 2.计算注意力权重
        # 权重形状 (N,L_dec,L_enc) --每个时间步都有一个长度为L_enc的注意力权重
        attention_weights = torch.softmax(attention_scores,dim=-1)
        # 3.计算上下文向量
        context_vector = torch.bmm(attention_weights,encoder_outputs)
        # 上下文向量形状：(N,L,hidden_size)
        return context_vector


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
        return output,features
#自定义解码器
class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层，指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True)
        # Attention层
        self.attention = Attention()
        #全连接层：
        self.linear = nn.Linear(in_features=HIDDEN_SIZE * 2,out_features=vocab_size)
    #前向传播,传入初始隐状态
    def forward(self, x,h0,encoder_output):
        # 1.词嵌入
        embed = self.embedding(x)
        # 2. x与h0在GRU前向传播，得到输出隐状态(N，L，hidden_size)--没有填充且为单层，所以输出的隐状态就是特征向量
        # gru输入格式(batch, seq_len, embedding_dim)
        output, hn = self.gru(embed,h0)
        # 3.注意力机制，基于解码器输出隐状态(N，L_dec，hidden_size)和编码器的每个时间步输出隐状态(N，L_enc，hidden_size)，计算动态上下文向量
        # 上下文向量形状：(N，L_dec，hidden_size)
        context_vecoter = self.attention(output,encoder_output)
        # 4. 将动态上下文向量拼接融合到隐状态信息中
        # 合并后特征向量形状：(N，L_dec，hidden_size*2)
        combined = torch.cat([context_vecoter,output],dim=-1)
        # 5. 整合特征，预测分类,这里只需保证最后最后一维能和linear对上就行，如果展开对应一定是又深意的。
        output = self.linear(combined)
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