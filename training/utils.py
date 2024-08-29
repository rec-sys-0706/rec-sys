import time
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from typing import Literal

class EarlyStopping:
    """EarlyStopping references tensorflow.keras.callbacks.EarlyStopping."""
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop_training = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            (stop_training, is_better) = (False, True)
        else:
            self.counter += 1
            (stop_training, is_better) = (False, False)
            if self.counter >= self.patience:
                (stop_training, is_better) = (True, False)
                self.stop_training = True

        return stop_training, is_better

def time_since(base: float, format: None|Literal['seconds']=None):
    now = time.time()
    elapsed_time = now - base
    if format == 'seconds':
        return elapsed_time
    else:
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

def detokenize(word2int_path, seq):
    seq = np.array(seq).astype(str)
    word2int = dict(pd.read_csv(word2int_path,
                                sep='\t',
                                names=['word', 'int'],
                                index_col=False).values)
    int2word = {v: k for k, v in word2int.items()}

    decode_map = np.vectorize(int2word.get)
    return decode_map(seq)

def get_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")

def format_duration(sec):
    return time.strftime("%H:%M:%S", time.gmtime(sec))
    
def fix_all_seeds(seed):
    '''Fixes RNG seeds for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print(get_now())