import logging
import time
from pathlib import Path
import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.NRMS import NRMS
from parameters import Arguments, parse_args
from utils import CustomTokenizer, EarlyStopping, time_since, get_datetime_now, fix_all_seeds
from dataset import NewsDataset
from evaluate import nDCG, ROC_AUC, recall, accuracy


evaluate = lambda _pred, _true: (recall(_pred, _true), ROC_AUC(_pred, _true), nDCG(_pred, _true, 5), nDCG(_pred, _true, 10), accuracy(_pred, _true))


def train(args: Arguments):
    """Trainer"""
    fix_all_seeds(args.seed)
    writer = SummaryWriter(
        log_dir=f"./runs/{get_datetime_now()}"
    )
    if not Path(args.ckpt_dir).exists():
        Path(args.ckpt_dir).mkdir()
    ckpt_now_dir = Path(args.ckpt_dir) / get_datetime_now() 
    ckpt_now_dir.mkdir()

    tokenizer = CustomTokenizer(args)
    model = NRMS(args=args, vocab_size=tokenizer.vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Prepare Datasets
    logging.info(f'Loading datasets...')
    start_time = time.time()
    train_dataset = NewsDataset(args, tokenizer, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)

    valid_dataset = NewsDataset(args, tokenizer, mode='valid')
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              pin_memory=True)
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')

    # Training Loop
    early_stopping = EarlyStopping(patience=args.patience)
    step = 0 # A counter to count model updated times.

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):

        train_loss = 0.0
        for batch in tqdm(train_loader, desc='Training'):
            model.train()
            y_pred = model(batch['clicked_news'], batch['candidate_news'])
            y_true = batch['clicked'].to(args.device)
            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()


            step += 1
            writer.add_scalar('Train/Loss', loss.item(), step)
            if args.valid_timing == 'interval' and step % args.valid_interval == 0:
                tqdm.write(f'Starting validation process in step {step:5}.')
                model.eval()
                y_preds = []
                y_trues = []
                valid_loss = 0.0
                for batch in tqdm(valid_loader, desc='Validation', leave=False):
                    y_pred = model(batch['clicked_news'], batch['candidate_news'])
                    y_true = batch['clicked'].to(args.device)
                    loss = criterion(y_pred, y_true)
                    valid_loss += loss.item()
                    
                    y_preds.append(y_pred.detach().cpu())
                    y_trues.append(y_true.cpu())
                valid_loss /= len(valid_loader)
                y_preds = torch.concat(y_preds)
                y_trues = torch.concat(y_trues)
                recall, roc_auc, ndcg5, ndcg10, acc = evaluate(y_preds, y_trues)
                stop_training, is_better = early_stopping(-recall) # TODO
                # Logging - valid
                tqdm.write((
                    f'Elapsed Time: {time_since(start_time)}\n'
                    f'Valid/Best Recall: {-early_stopping.best_loss:6.4f}\n'
                    f'Valid/Recall: {recall:6.4f}\n'
                    f'Valid/ROC_AUC: {roc_auc:6.4f}\n'
                    f'Valid/nDCG@5: {ndcg5:6.4f}\n'
                    f'Valid/nDCG@10: {ndcg10:6.4f}\n'
                    f'Valid/Accuracy: {acc:6.4f}\n'
                ))
                writer.add_scalar('Valid/Recall', recall, step)
                writer.add_scalar('Valid/ROC_AUC', roc_auc, step)
                writer.add_scalar('Valid/nDCG@5', ndcg5, step)
                writer.add_scalar('Valid/nDCG@10', ndcg10, step)
                writer.add_scalar('Valid/Accuracy', acc, step)

                if is_better:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch},  ckpt_now_dir / f'ckpt-{step}.pth')
                    tqdm.write('Model performance improved, model is saved.')
        
        
        
        if early_stopping.stop_training:
            break
        train_loss /= len(train_loader)
        tqdm.write((
            f'Epoch: {epoch:3}/{args.epochs} Elapsed Time: {time_since(start_time)}\n'
            f'Train/Loss: {train_loss}\n'
        ))
        writer.add_scalar('Train/Loss', loss, epoch)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch},  ckpt_now_dir / f'ckpt-latest.pth') # TODO checkpoint folder
    tqdm.write((
        f'Training process ended at Epoch {epoch:3}/{args.epochs}\n'
        f'Valid/Best Recall: {-early_stopping.best_loss:6.4f}\n'
    ))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    start = time.time()
    args = parse_args()
    train(args)


    # tokenizer = CustomTokenizer(args)
    # dataset = NewsDataset(args, tokenizer, 'train')
    # data_collator = CustomDataCollator()
    # # Create DataLoader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     collate_fn=data_collator,
    # )
    # print(time_since(start))
    # # Iterate over DataLoader and print batch shapes
    # for _ in range(10):
    #     for batch in dataloader:
    #         pass
    #         # print(batch['clicked_news_title']['input_ids'].shape)
    #         # print(batch['candidate_news_title']['input_ids'].shape)
    #         # print(batch['labels'].shape)
    #     print(time_since(start))



# args = TrainingArguments(
#     Path(CONFIG.ckpt_dir) / f'ckpt-{step}.pth',
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
#     push_to_hub=True,
# )
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     push_to_hub=True,
# )