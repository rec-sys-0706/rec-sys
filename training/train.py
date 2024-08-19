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
from evaluate import nDCG, MRR, ROC_AUC
from utils import EarlyStopping

CONFIG = BaseConfig()
try:
    Model = getattr(importlib.import_module(f'model.{CONFIG.model_name}'), CONFIG.model_name)
except AttributeError:
    raise AttributeError(f'{CONFIG.model_name} not included!')
except ImportError:
    raise ImportError(f"Error importing {CONFIG.model_name} model.")

# TODO early stopping
# TODO checkpoint
# TODO evaluate
# TODO time count
def train():
    """Training loop"""
    writer = SummaryWriter(
        log_dir=f"./runs/"
    )
    device = CONFIG.device
    model: nn.Module = Model(CONFIG, device).to(device)
    # TODO load pretrain
    # model = Model(config, pretrained_word_embedding,
    #                 pretrained_entity_embedding,
    #                 pretrained_context_embedding).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # TODO optimizer

    # training loop
    logging.info(f'Preparing datasets...')
    train_dataset = NewsDataset(CONFIG, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size) # TODO shuffle

    valid_dataset = NewsDataset(CONFIG, mode='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size)

    early_stopping = EarlyStopping(patience=3)


    # TODO logging something.
    # TODO delete logging.info(f'Starting training process with {CONFIG.max_epochs} epochs.')
    step = 0
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

            train_loss += item(loss)


            step += 1
            if step % CONFIG.valid_interval == 0:
                logging.info(f'Starting validation process in step {step:5}.')
                model.eval()
                evaluate = lambda _pred, _true: (0, 0, nDCG(_pred, _true, 5), nDCG(_pred, _true, 10))
                y_preds = []
                y_trues = []
                valid_loss = 0.0
                for item in tqdm(valid_loader, desc='Validation', leave=False):
                    y_pred = model(item['clicked_news'], item['candidate_news'])
                    y_true = item['clicked']
                    loss = criterion(y_pred, y_true)
                    valid_loss += loss.item()
                    
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(y_true)
                valid_loss /= len(valid_loader)
                y_preds = torch.concat(y_preds)
                y_trues = torch.concat(y_trues)
                roc_auc, mrr, ndcg5, ndcg10 = evaluate(y_preds, y_trues)
                print((
                    f'\nValid/ROC_AUC: {0.0:6.4}\n',
                    f'Valid/MRR: {0.0:6.4}\n'
                    f'Valid/nDCG@5: {ndcg5:6.4}\n'
                    f'Valid/nDCG@10: {ndcg10:6.4}\n'
                    f'Valid/Loss: {valid_loss}\n'
                ))
                stop_training, is_better = early_stopping(valid_loss)

                if stop_training:
                    pass # TODO
                elif is_better:
                    pass # TODO
        train_loss /= len(train_loader)
        logging.info((
            f'Epoch: {epoch:3}/{CONFIG.max_epochs}\n'
            f'Train/Loss: {train_loss}\n'
        ))
        writer.add_scalar('Train/Loss', loss, epoch)


    # TODO model
    # writer.add_scalar('Loss/test', np.random.random(), step)
    # writer.add_scalar('Accuracy/train', np.random.random(), step)
    # writer.add_scalar('Accuracy/test', np.random.random(), step)
    # writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    train()