from pathlib import Path
# 1. 目录路径
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
# 数据目录
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'
# 模型目录
MODEL_DIR = ROOT_DIR / 'models'
##日志目录
LOG_DIR = ROOT_DIR/'logs'
# 2. 文件
RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
# VOCAB_FILE = 'vocab.txt'    # 词表文件
EN_VOCAB_FILE = 'en_vocab.txt'    # 英文词表
CN_VOCAB_FILE = 'cn_vocab.txt'    # 中文词表
BEST_MODEL = 'best_model.pt'    # 最优模型参数文件

# 3. 特殊token
UNK_TOKEN = '<UNK>'     # 未登录词
PAD_TOKEN = '<pad>'     # 填 充 词
START_TOKEN = '<sos>'  #起始标记
END_TOKEN = '<eos>'    #结束标记
# 4. 训练超参数
SEQ_LEN = 128     # 序列（最大）长度
BATCH_SIZE = 64

LEARNING_RATE = 1e-3
EPOCHS = 50

# 5. 模型结构参数
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
