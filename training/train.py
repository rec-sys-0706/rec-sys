import importlib
from config import BaseConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NewsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb
import logging
from evaluate import nDCG, ROC_AUC
from utils import EarlyStopping, time_since
from pathlib import Path
import time

CONFIG = BaseConfig()
try:
    Model = getattr(importlib.import_module(f'model.{CONFIG.model_name}'), CONFIG.model_name)
except AttributeError:
    raise AttributeError(f'{CONFIG.model_name} not included!')
except ImportError:
    raise ImportError(f"Error importing {CONFIG.model_name} model.")


def train():
    """Training loop"""
    writer = SummaryWriter(
        log_dir=f"./runs/"
    )
    if not Path(CONFIG.ckpt_dir).exists():
        Path(CONFIG.ckpt_dir).mkdir()
    device = CONFIG.device
    model: nn.Module = Model(CONFIG, device).to(device)
    # TODO load pretrain
    # model = Model(config, pretrained_word_embedding,
    #                 pretrained_entity_embedding,
    #                 pretrained_context_embedding).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    # Training Loop
    logging.info(f'Preparing datasets...')
    train_dataset = NewsDataset(CONFIG, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG.train_batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    valid_dataset = NewsDataset(CONFIG, mode='valid')
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CONFIG.valid_batch_size,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True)

    early_stopping = EarlyStopping(patience=3)

    step = 0
    start_time = time.time()
    for epoch in range(1, CONFIG.max_epochs + 1):

        train_loss = 0.0
        model.train()
        for item in tqdm(train_loader, desc='Training'):
            y_pred = model(item['clicked_news'], item['candidate_news'])
            y_true = item['clicked'].to(device)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()


            step += 1
            if step % CONFIG.valid_interval == 0:
                tqdm.write(f'Starting validation process in step {step:5}.')
                model.eval()
                evaluate = lambda _pred, _true: (ROC_AUC(_pred, _true), nDCG(_pred, _true, 5), nDCG(_pred, _true, 10))
                y_preds = []
                y_trues = []
                valid_loss = 0.0
                for item in tqdm(valid_loader, desc='Validation', leave=False):
                    y_pred = model(item['clicked_news'], item['candidate_news'])
                    y_true = item['clicked'].to(device)
                    loss = criterion(y_pred, y_true)
                    valid_loss += loss.item()
                    
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(y_true.cpu())
                valid_loss /= len(valid_loader)
                y_preds = torch.concat(y_preds)
                y_trues = torch.concat(y_trues)
                roc_auc, ndcg5, ndcg10 = evaluate(y_preds, y_trues)
                # Logging - valid
                tqdm.write((
                    f'Elapsed Time: {time_since(start_time)}\n'
                    f'Valid/ROC_AUC: {roc_auc:6.4}\n'
                    f'Valid/nDCG@5: {ndcg5:6.4}\n'
                    f'Valid/nDCG@10: {ndcg10:6.4}'
                ))
                writer.add_scalar('Valid/ROC_AUC', roc_auc, step)
                writer.add_scalar('Valid/nDCG@5', ndcg5, step)
                writer.add_scalar('Valid/nDCG@10', ndcg10, step)
                writer.add_scalar('Valid/nDCG@10', ndcg10, step)

                stop_training, is_better = early_stopping(-roc_auc)
                if stop_training:
                    break
                elif is_better:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch},  Path(CONFIG.ckpt_dir) / f'ckpt-{step}.pth')

        if early_stopping.stop_training:
            tqdm.write((
                f'Training process ended at Epoch {epoch}\n'
                f'Best Valid/Loss: {early_stopping.best_loss}\n'
            ))
        train_loss /= len(train_loader)
        tqdm.write((
            f'Epoch: {epoch:3}/{CONFIG.max_epochs} Elapsed Time: {time_since(start_time)}\n'
            f'Train/Loss: {train_loss}\n'
        ))
        writer.add_scalar('Train/Loss', loss, epoch)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    train()