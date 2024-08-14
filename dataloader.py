import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *
from data_utils import pre_process_data

class StockData(Dataset):
    def __init__(self, df, SEQ_LEN, PRED_LEN, INPUT_COLS, OUTPUT_COLS):
        self.df = df
        self.seq_length = SEQ_LEN
        self.pred_length = PRED_LEN
        self.input_cols = INPUT_COLS
        self.output_cols = OUTPUT_COLS
            
    def __len__(self):
        return len(self.df) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        x = self.df.iloc[idx:idx+self.seq_length][self.input_cols].values
        # x = x.reshape(-1, 1, len(self.__input_cols))
        x = torch.from_numpy(x).float()
        y = self.df.iloc[idx+self.seq_length:idx+self.seq_length+self.pred_length][self.output_cols].values
        # y = y.reshape(-1, 1, len(self.__output_cols))
        y = torch.from_numpy(y).float()
        
        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

if __name__ == '__main__':
    # Prints a sample of the data
    # Read the data
    df = pd.read_csv(FILE_PATH, sep=',', index_col=False, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Pre-process the data
    df_resampled = pre_process_data(df)
    print(df_resampled.head(-1))
    print(df_resampled.info())
    
    dataset = StockData(df_resampled, SEQ_LEN=5, PRED_LEN=2, INPUT_COLS=['close', 'open', 'RSI'], OUTPUT_COLS=['close'])
    # print("Length", len(dataset))
    x, y = dataset[60]
    print("Sample")
    print(x)
    print(y)
    