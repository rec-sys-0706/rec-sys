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

CONFIG = BaseConfig()
try:
    Model = getattr(importlib.import_module(f'model.{CONFIG.model_name}'), CONFIG.model_name)
except AttributeError:
    raise AttributeError(f'{CONFIG.model_name} not included!')
except ImportError:
    raise ImportError(f"Error importing {CONFIG.model_name} model.")

# TODO early stopping
def train():
    """Training loop"""
    writer = SummaryWriter(
        log_dir=f"./runs/"
    )
    device = CONFIG.device
    model: nn.Module = Model(CONFIG, device).to(device)
    # model = Model(config, pretrained_word_embedding,
    #                 pretrained_entity_embedding,
    #                 pretrained_context_embedding).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # TODO optimizer

    # training loop
    dataset = NewsDataset(CONFIG, 'data/MINDsmall_train/behaviors_parsed.tsv', 'data/MINDsmall_train/news_parsed.tsv')
    loader = DataLoader(dataset, batch_size=32) # TODO shuffle

    # TODO logging something.
    for epoch in tqdm(CONFIG.max_epochs):
        losses = []
        for item in tqdm(loader):
            y_pred = model(item['clicked_news'], item['candidate_news'])
            y_true = item['clicked'].to(device)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        mean_train_loss = sum(losses) / len(losses)
        print(f'Epoch: {epoch}, Loss/trainL: {mean_train_loss}')
        writer.add_scalar('Loss/train', mean_train_loss, epoch)
    # TODO model
    # writer.add_scalar('Loss/test', np.random.random(), step)
    # writer.add_scalar('Accuracy/train', np.random.random(), step)
    # writer.add_scalar('Accuracy/test', np.random.random(), step)
    # writer.close()


if __name__ == '__main__':
    train()
    exit()
    for i in dataloader:
        i: dict
        print(i)
        break

    for i in range(100):
        break
        y_pred = model()
        y_true = ...
        loss = criterion(y_pred, y_true)
        