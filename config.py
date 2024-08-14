import os

# Files
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SYMBOL = 'CC'
FILE_NAME = 'CC_1min.txt'
FILE_PATH = DIR_PATH + '\data\stock\\' + FILE_NAME

# Problem definition
SEQ_LEN = 35 # Number of days of historical data used to predict
PRED_LEN = 7 # Number of days to predict

# Feature engineering
# Available features: ['close', 'open', 'high', 'low', 'volume', 'ema_5', 'ema_10', 'ema_20', 'RSI', 'day_of_week']
# 'close' MUST BE the FIRST element in both input and output columns
INPUT_COLS = ['close', 'open', 'ema_5', 'ema_10', 'RSI', 'day_of_week']
OUTPUT_COLS = ['close']
BATCH_SIZE = 48

# Model architecture
INPUT_SIZE = len(INPUT_COLS)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = len(OUTPUT_COLS)
DROP_OUT = 0.2

# Training
DEVICE = 'cuda'
EPOCHS = 40
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# test ratio is 1 - (TRAIN_RATIO + VAL_RATIO), no need to define it

# Stock Simulation Paramerters
INITIAL_CASH = 100 # Starting cash
TRADE_COMMISION = 0.001 # Transaction cost as a fraction of the total transaction amount