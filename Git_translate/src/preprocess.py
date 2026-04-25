from config import *
import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer,EnglishTokenizer

def preprocess():
    print('------------------开始数据预处理-------------------')
    #1.以csv格式读取txt文件，得到DataFrame；并提取两列，
    # (要点：指明分隔符是制表符，默认excel文件分隔符是‘,’,txt中没有列名，用usecols=['列号']来读可以用name加上)
    df = pd.read_csv(RAW_DATA_DIR/RAW_DATA_FILE,sep='\t',usecols=[0,1], names=['en','cn'], encoding='utf-8').dropna()
    #2. 对原始语料做划分
    train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
    #3.分词并构建词表，保存到文件(build_vocab读的是列表)
    #调用逻辑：先调用类方法中构建词表到model文件下，方便下一步对对应词表构建相应的分词器
    ChineseTokenizer.build_vocab(train_df['cn'].tolist(),MODEL_DIR/CN_VOCAB_FILE)
    EnglishTokenizer.build_vocab(train_df['en'].tolist(),MODEL_DIR/EN_VOCAB_FILE)
    #4.创建分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)
    #5.构建数据集
    train_df['cn'] = train_df['cn'].apply(lambda text:cn_tokenizer.encode(text,mark=False))
    train_df['en'] = train_df['en'].apply(lambda text:en_tokenizer.encode(text,mark=True))

    test_df['cn'] = test_df['cn'].apply(lambda text:cn_tokenizer.encode(text,mark=False))
    test_df['en'] = test_df['en'].apply(lambda text:en_tokenizer.encode(text,mark=True))

    #6.保存数据到文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE,orient='records',lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE, orient='records', lines=True)

    print("-------数据预处理完成-------")

if __name__ == '__main__':
    preprocess()
