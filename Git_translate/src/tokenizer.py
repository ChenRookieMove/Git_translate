import jieba
from config import *
from nltk import TreebankWordTokenizer,TreebankWordDetokenizer  # 英文分词器，合并器

#分词器的父类，中英文分词都需要的共同前置步骤
class BaseTokenizer():
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN
    start_token  = START_TOKEN
    end_token = END_TOKEN
    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2id = {word:id for id,word in enumerate(vocab_list)}
        self.id2word = {id:word for id,word in enumerate(vocab_list)}
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]
        self.start_id = self.word2id[self.start_token]
        self.end_id = self.word2id[self.end_token]
    # 分词，类方法接口
    # 父类 Object-Oriented Programming---BaseTokenizer 先规定：
    # 所有分词器都必须有一个 tokenize() 方法。
    # 也就是说父类定义“规范”，子类负责实现。
    # 优化点：可以加上报错：如果子类没有tokenize反而调用了父类的tokenize会有提示
    @classmethod
    def tokenize(cls,text) -> list[str]:
        pass
    # 编码（将文本分词，id化）
    def encode(self,text,mark= False):
        #疑问：再这里加入mark选项，再什么情况下，这里的源语音不需要标记开头和结尾；还是说这里的encode是encoder和decoder共用的。
        #回答：这里的encode函数是编码器和解码器公用的，对于编码器而言其中的输入句子是固定的，无需sos和eos。
        tokens = self.tokenize(text)
        #填充或截断到指定长度(训练时的batch，填充到同一长度才能形成张量，批量计算)
        if mark:
            tokens = [self.start_token] + tokens + [self.end_token]
        ids = [self.word2id.get(token,self.unk_id) for token in tokens]
        return ids
    #构建词表,并保存到文件（该方法要调用类方法中的tokenize方法，所以传入cls）
    @classmethod
    def build_vocab(cls,sentences,vocab_file_path):
        vocab_set = set()
        for sentence in sentences:
            vocab_set.update(cls.tokenize(sentence))
        vocab_list = [cls.pad_token,cls.unk_token,cls.start_token,cls.end_token] + list(vocab_set)
        print(f'词表大小：{len(vocab_list)}')
        with open(vocab_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    ##（工厂方法）从文件记载词表，并创建分词器对象实例
    @classmethod
    def from_vocab(cls,vocab_file_path):
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab_list = [token.strip() for token in f.readlines()]
        ##构建分词器对象
        tokenizer = cls(vocab_list)
        return tokenizer
# 定义子类
# 中文分词器
class ChineseTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls,text) -> list[str]:
        return list(text)
# 英文分词器,其中带解码方法
class EnglishTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    @classmethod
    def tokenize(cls,text) -> list[str]:
        return cls.tokenizer.tokenize(text)
    # 解码，传入一个id列表，返回原始的英文句子
    def decode(self,ids):
        #将id转换为tokens
        tokens = [self.id2word[id] for id in ids]
        return self.detokenizer.detokenize(tokens)
if __name__ == '__main__':
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR/EN_VOCAB_FILE)
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR/CN_VOCAB_FILE)


    print('特殊符号UNK',en_tokenizer.unk_token)
    print('特殊符号PAD ID',cn_tokenizer.pad_id)
    print(cn_tokenizer.encode('自然语言处理'))
    print(en_tokenizer.encode('Hello world!',mark=True))
