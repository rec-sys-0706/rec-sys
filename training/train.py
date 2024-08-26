import importlib
from config import BaseConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import NewsDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from evaluate import nDCG, ROC_AUC
from utils import EarlyStopping, time_since, detokenize, get_now
from pathlib import Path
import time
import pandas as pd

CONFIG = BaseConfig()
try:
    Model = getattr(importlib.import_module(f'model.{CONFIG.model_name}'), CONFIG.model_name)
except AttributeError:
    raise AttributeError(f'{CONFIG.model_name} not included!')
except ImportError:
    raise ImportError(f"Error importing {CONFIG.model_name} model.")

evaluate = lambda _pred, _true: (ROC_AUC(_pred, _true), nDCG(_pred, _true, 5), nDCG(_pred, _true, 10))

def train():
    """Training loop"""
    
    writer = SummaryWriter(
        log_dir=f"./runs/{get_now()}"
    )
    if not Path(CONFIG.ckpt_dir).exists():
        Path(CONFIG.ckpt_dir).mkdir()


    if CONFIG.use_pretrained_embedding:
        pretrained_embedding = torch.load(Path(CONFIG.train_dir) / 'pretrained_embedding.pt')
        if pretrained_embedding.shape != (CONFIG.vocab_size, CONFIG.embedding_dim):
            raise ValueError((
                "Error: Expected pretrained embeddings to have the same shape as (config.vocab_size, config.embedding_dim), "
                f"but got {pretrained_embedding.shape} and ({CONFIG.vocab_size}, {CONFIG.embedding_dim})."
                "Please regenerating pretrained word embeddings."
            ))
        model: nn.Module = Model(CONFIG, pretrained_embedding)
    else:
        model: nn.Module = Model(CONFIG)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    # Training Loop
    logging.info(f'Loading datasets...')
    start_time = time.time()
    train_dataset = NewsDataset(CONFIG, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=CONFIG.train_batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    valid_dataset = NewsDataset(CONFIG, mode='valid')
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CONFIG.valid_batch_size,
                              pin_memory=True)
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')

    early_stopping = EarlyStopping(patience=CONFIG.patience)

    step = 0
    start_time = time.time()
    for epoch in range(1, CONFIG.max_epochs + 1):

        train_loss = 0.0
        model.train()
        for item in tqdm(train_loader, desc='Training'):
            y_pred = model(item['clicked_news'], item['candidate_news'])
            y_true = item['clicked'].to(CONFIG.device)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()


            step += 1
            if step % CONFIG.valid_interval == 0:
                tqdm.write(f'Starting validation process in step {step:5}.')
                model.eval()
                y_preds = []
                y_trues = []
                valid_loss = 0.0
                for item in tqdm(valid_loader, desc='Validation', leave=False):
                    y_pred = model(item['clicked_news'], item['candidate_news'])
                    y_true = item['clicked'].to(CONFIG.device)
                    loss = criterion(y_pred, y_true)
                    valid_loss += loss.item()
                    
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(y_true.cpu())
                valid_loss /= len(valid_loader)
                y_preds = torch.concat(y_preds)
                y_trues = torch.concat(y_trues)
                roc_auc, ndcg5, ndcg10 = evaluate(y_preds, y_trues)
                stop_training, is_better = early_stopping(-roc_auc)
                # Logging - valid
                tqdm.write((
                    f'Elapsed Time: {time_since(start_time)}\n'
                    f'Best Valid/ROC_AUC: {-early_stopping.best_loss:6.4f}\n'
                    f'Valid/ROC_AUC: {roc_auc:6.4f}\n'
                    f'Valid/nDCG@5: {ndcg5:6.4f}\n'
                    f'Valid/nDCG@10: {ndcg10:6.4f}'
                ))
                writer.add_scalar('Valid/ROC_AUC', roc_auc, step)
                writer.add_scalar('Valid/nDCG@5', ndcg5, step)
                writer.add_scalar('Valid/nDCG@10', ndcg10, step)
                writer.add_scalar('Valid/nDCG@10', ndcg10, step)

                if stop_training:
                    break
                elif is_better:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch},  Path(CONFIG.ckpt_dir) / f'ckpt-{step}.pth')
        if early_stopping.stop_training:
            break
        train_loss /= len(train_loader)
        tqdm.write((
            f'Epoch: {epoch:3}/{CONFIG.max_epochs} Elapsed Time: {time_since(start_time)}\n'
            f'Train/Loss: {train_loss}\n'
        ))
        writer.add_scalar('Train/Loss', loss, epoch)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch},  Path(CONFIG.ckpt_dir) / f'ckpt-latest.pth') # TODO checkpoint folder
    tqdm.write((
        f'Training process ended at Epoch {epoch:3}/{CONFIG.max_epochs}\n'
        f'Best Valid/Loss: {-early_stopping.best_loss:6.4f}\n'
    ))

def test():
    model: nn.Module = Model(CONFIG)
    checkpoint = torch.load(Path(CONFIG.ckpt_dir) / 'ckpt-latest.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test loop
    model.eval()
    test_dataset = NewsDataset(CONFIG, mode='test')
    test_loader = DataLoader(test_dataset,
                             batch_size=CONFIG.valid_batch_size,
                             pin_memory=True)
    users_iter = iter(test_dataset.users)

    y_preds = []
    users = []
    for item in tqdm(test_loader, desc='Testing', leave=False):
        y_pred = model(item['clicked_news'], item['candidate_news'])
        y_preds.append(y_pred.detach().cpu())

        clickeds = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0)).tolist()

        for clicked in clickeds:
            user = next(users_iter)
            users.append({
                'user_id': user['user_id'],
                'clicked_news': user['clicked_news_ids'],
                'candidate_news': user['candidate_news_ids'],
                'clicked': clicked
            })
    y_preds = torch.concat(y_preds)
    # Save
    (pd.DataFrame(users, columns=['user_id', 'clicked_news', 'candidate_news', 'clicked'])
     .to_csv(Path(CONFIG.test_dir) / 'result.csv',
             index=False))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    train()
    # test()
    # TODO fix seed
    # TODO pyplot
    # TODO recall metrics