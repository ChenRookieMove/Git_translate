import time
import torch
from torch import nn,optim
from tqdm import tqdm
from config import *
from dataset import get_dataloader
from model import TranslationModel
from torch.utils.tensorboard import SummaryWriter
from tokenizer import ChineseTokenizer,EnglishTokenizer

def train_one_epoch(model,train_loader,loss,optimizer,device):
    model.train()
    total_loss = 0
    # 按批次进行迭代
    for inputs,targets in tqdm(train_loader,desc='训练：'):
        inputs,targets = inputs.to(device),targets.to(device) ###形状(N=64,L)
        #1.前向传播
        #1.1 编码,得到一批上下文向量(N,hidden_size)
        context_vectors = model.encoder(inputs)
        #1.2 解码,Teacher Forcing
        #得到解码的输入和目标
        #看似不符合GRU的输入格式，实则在decoder中有一层embadding层让其先加了个维度
        decoder_inputs = targets[:,:-1]
        decoder_targets = targets[:,1:]
        #用编码得到的上下文向量，作为初始隐状态，形状(1,N,hidden_size)
        decoder_h0 = context_vectors.unsqueeze(0)  #变得符合输入维度
        #解码器前向传播，得到解码输出，(N,L,vocab_size)
        decoder_outputs,_ = model.decoder(decoder_inputs,decoder_h0)
        #2.计算损失,输出形状(N,L,vocab_size),目标形状(N,L)
        # 但是需要调整输出维度为(N,vocab_size,L)
        loss_value =loss(decoder_outputs.transpose(1,2),decoder_targets)
        #3.反向传播
        loss_value.backward()
        #4.更新参数
        optimizer.step()
        #5.梯度清零
        optimizer.zero_grad()
        #累加损失
        total_loss+=loss_value.item()
    return total_loss/len(train_loader)
def train():
    #1.定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #2.创建数据加载器
    train_loader = get_dataloader()
    #3。获取词表，创建分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)
    #4.定义模型
    model = TranslationModel(cn_tokenizer.vocab_size,en_tokenizer.vocab_size,cn_tokenizer.pad_id,en_tokenizer.pad_id).to(device)
    #5.定义损失函数(不计算pad的损失)
    loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
    #6.定义优化方法
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    #7.定义一个tensorboard写入器
    writer = SummaryWriter(log_dir = LOG_DIR / time.strftime('%Y-%m-%d-%H%M-%S'))
    #8.核心训练流程，按照epoch进行迭代
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*10,f'epoch:{epoch+1}','='*10)
        this_loss = train_one_epoch(model,train_loader,loss,optimizer,device)
        print('本轮训练损失：',this_loss)
        writer.add_scalar('loss',this_loss,epoch+1)

        if this_loss<min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(),MODEL_DIR/BEST_MODEL)
            print('模型保存成功')
    writer.close()
if __name__ == '__main__':
    train()