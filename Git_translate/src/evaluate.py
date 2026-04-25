import torch
from tqdm import tqdm
from config import *
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer,EnglishTokenizer
from nltk.translate.bleu_score import corpus_bleu # 引入评价指标



def evaluate(model,dataloader,tokenizer,device):
    # 用列表记录参考译文和预测译文
    references = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs,targets in tqdm(dataloader, desc='评估：'):
            inputs = inputs.to(device)
            # 转成列表，下面才能用index方法，这里由于后续说inputs并行计算，targets也不涉及前向传播等并行计算，所以可以直接变成列表
            targets = targets.tolist()
            # 前向传播，得到一批样本的预测译文
            batch_reslut = predict_batch(model,inputs,tokenizer,device)
            # 合并这一批结果到总列表
            predictions.extend(batch_reslut)
            # 合并这一批目标值(参考译文)到总列表
            # BLEU 的 reference 支持“一条预测对应多个参考答案”,所以它要求比 prediction 多套一层。
            references.extend([[target[1:target.index(tokenizer.end_id)]] for target in targets])
    # 调库计算bleu评分
    bleu_score = corpus_bleu(references,predictions)
    return bleu_score

def run_evaluate():
    # 1. 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2. 获取词表
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)
    print('词表加载成功')
    # 3. 加载模型
    model = TranslationModel(cn_tokenizer.vocab_size,en_tokenizer.vocab_size,cn_tokenizer.pad_id,en_tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR/BEST_MODEL))
    print('模型加载成功')
    # 4. 获取测试集数据
    test_dataloader =get_dataloader(train=False)
    # 5. 调用评分逻辑
    bleu = evaluate(model,test_dataloader,en_tokenizer,device)
    print('评估结果：')
    print('BLEU评分:',bleu)
if __name__ == '__main__':
    run_evaluate()