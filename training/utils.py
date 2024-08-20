import time
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

def time_since(base: float):
    now = time.time()
    elapsed_time = now - base
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))