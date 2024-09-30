import logging
import time
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.data._utils.collate import default_collate
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file

from data_preprocessing import data_preprocessing
from model.NRMS import NRMS, NRMS_BERT
from parameters import Arguments, parse_args
from utils import CustomTokenizer, time_since, get_datetime_now, fix_all_seeds, traverse_object
from dataset import NewsDataset, CustomDataCollator
from evaluate import nDCG, ROC_AUC, recall, accuracy
evaluate = lambda _pred, _true: (recall(_pred, _true), ROC_AUC(_pred, _true), nDCG(_pred, _true, 5), nDCG(_pred, _true, 10), accuracy(_pred, _true))


def compute_metrics(eval_preds):
    predictions, label_ids = eval_preds
    recall, auc, ndcg5, ndcg10, acc = evaluate(predictions, label_ids)
    return {
        'recall': recall,
        'auc': auc,
        'ndcg5': ndcg5,
        'ndcg10': ndcg10,
        'acc': acc
    }

def get_model(args: Arguments, tokenizer: CustomTokenizer) -> nn.Module:
    if args.model_name == 'NRMS':
        model = NRMS(args=args, vocab_size=tokenizer.vocab_size)
    elif args.model_name == 'NRMS-Glove':
        pretrained_embedding = torch.load(Path(args.train_dir) / 'pretrained_embedding.pt')
        pretrained_embedding = pretrained_embedding.contiguous()
        if pretrained_embedding.shape != (tokenizer.vocab_size, args.embedding_dim):
            raise ValueError((
                "Error: Expected pretrained embeddings to have the same shape as (args.vocab_size, args.embedding_dim), "
                f"but got {pretrained_embedding.shape} and ({tokenizer.vocab_size}, {args.embedding_dim})."
                "Please regenerating pretrained word embeddings."
            ))
        model = NRMS(args=args,
                     vocab_size=tokenizer.vocab_size,
                     pretrained_embedding=pretrained_embedding)
    elif args.model_name == 'NRMS-BERT':
        model = NRMS_BERT(args=args,
                          pretrained_model_name=args.pretrained_model_name)
    else:
        raise ValueError(f"Model name `{args.model_name}` did not supported.")
    return model

def next_folder(folder_name):
    # Check if the `folder_name` ends with _##, where ## are digits.
    match = re.search(r'_(\d{2})$', folder_name)
    if match:
        number = int(match.group(1)) + 1
        next_number = f'{number:02d}'
        next_folder_name = re.sub(r'_(\d{2})$', f'_{next_number}', folder_name)
    else:
        next_folder_name = f'{folder_name}_01'

    if Path(f'checkpoints/{next_folder_name}').exists():
        return next_folder(next_folder_name)
    return next_folder_name
def get_ckpt_dir(args: Arguments):
    """Get log_dir and ckpt_dir by using Datatime"""
    folder_name = args.ckpt_dir
    if folder_name is not None:
        if not Path(f'checkpoints/{folder_name}').exists():
            raise ValueError(f"checkpoint `{folder_name}` did not exist in ./checkpoints.")
        folder_name = next_folder(folder_name)
        ckpt_dir = f'checkpoints/{folder_name}'
    else:
        DATETIME_NOW = get_datetime_now()
        ckpt_dir = f'checkpoints/{DATETIME_NOW}'
        if not Path('checkpoints').exists():
            Path('checkpoints').mkdir()

    Path(ckpt_dir).mkdir()
    with open(f'{ckpt_dir}/args.json', 'w') as f:
        args_dict = {k: str(v) for k, v in vars(args).items()}
        json.dump(args_dict, f)

    return ckpt_dir

def get_trainer_args(args: Arguments, ckpt_dir) -> TrainingArguments:
    return TrainingArguments(
        ckpt_dir,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_strategy='epoch',
        num_train_epochs=args.epochs,
        label_names=['clicked'],
        logging_dir=ckpt_dir, # TensorBoard
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        # greater_is_better='True',
        remove_unused_columns=False # If True, DataCollator will loss some info.
        
        # logging_steps=
        #logging_strategy=
        # optim=
        # warmup_steps=500
        # weight_decay=0.01
        # dataloader_drop_last=
        # push_to_hub=True,
    )  

def main(args: Arguments):
    next_ckpt_dir = get_ckpt_dir(args)
    trainer_args = get_trainer_args(args, next_ckpt_dir)
    # Tokenizer & Model & DataCollator
    tokenizer = CustomTokenizer(args)
    model = get_model(args, tokenizer)
    collate_fn = CustomDataCollator(tokenizer)
    # Prepare Datasets
    logging.info(f'Loading datasets...')
    start_time = time.time()
    if args.mode == 'train':
        train_dataset = NewsDataset(args, tokenizer, mode='train')
        valid_dataset = NewsDataset(args, tokenizer, mode='valid')
        test_dataset = None
    if args.mode == 'valid':
        train_dataset = None
        valid_dataset = NewsDataset(args, tokenizer, mode='valid')
        test_dataset = None
    if args.mode == 'test':
        train_dataset = None
        valid_dataset = None
        test_dataset = NewsDataset(args, tokenizer, mode='test')
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')
    # Trainer
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience,
    )
    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    # Load Model
    if (args.ckpt_dir is not None) and args.mode in ['valid', 'test']:
        last_checkpoint = get_last_checkpoint(f'checkpoints/{args.ckpt_dir}')
        tensors = load_file(last_checkpoint + "/model.safetensors")
        model_state_dict = model.state_dict()
        for key in model_state_dict.keys():
            if key in tensors:
                model_state_dict[key] = tensors[key]
        model.load_state_dict(model_state_dict)

    if args.mode == 'train':
        if args.ckpt_dir is None:
            trainer.train()
        else:
            pass
            # trainer.train(resume_from_checkpoint=f'checkpoints/{args.ckpt_dir}') # TODO continue training
    elif args.mode == 'valid':
        dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn)
        model.eval()
        predictions = []
        user_ids = []
        clicked_news_ids = []
        candidate_news_ids = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                predictions.append(outputs['logits'].cpu().numpy())
                labels.append(batch['clicked'].numpy())
                user = outputs['user']
                user_ids += user['user_id']
                clicked_news_ids += user['clicked_news_ids']
                candidate_news_ids += user['candidate_news_ids']
        predictions = np.concatenate(predictions, axis=0)
        result = np.where(predictions > 0.5, 1, 0).tolist()
        labels = np.concatenate(labels, axis=0).astype(int).tolist()
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'clicked_news': clicked_news_ids,
            'candidate_news': candidate_news_ids,
            'predictions': result,
            'labels': labels
        })
        df['clicked_news'] = df['clicked_news'].apply(lambda lst: [x for x in lst if x is not None])
        df['num_clicked_news'] = df['clicked_news'].apply(len)
        df = df.sort_values(by='user_id')
        df.to_csv(
            Path(next_ckpt_dir) / 'eval_result.csv',
            columns=['user_id',
                     'clicked_news',
                     'candidate_news',
                     'num_clicked_news',
                     'labels',
                     'predictions'],
            index=False
        )
        print(compute_metrics((predictions, labels)))
    elif args.mode == 'test':
        dataloader = DataLoader(test_dataset, batch_size=1)
        model.eval()
        predictions = []
        user_ids = []
        clicked_news_ids = []
        candidate_news_ids = []
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                predictions.append(outputs['logits'].cpu().numpy())
                user = outputs['user']
                user_ids += user['user_id']
                clicked_news_ids += user['clicked_news_ids']
                candidate_news_ids += user['candidate_news_ids']
        predictions = np.concatenate(predictions, axis=0)
        result = np.where(predictions > 0.5, 1, 0).tolist()
        
        df = pd.DataFrame({
            'user_id': user_ids,
            'clicked_news': clicked_news_ids,
            'candidate_news': candidate_news_ids,
            'predictions': result
        })
        df['clicked_news'] = df['clicked_news'].apply(lambda lst: [x for x in lst if x is not None])
        df['num_clicked_news'] = df['clicked_news'].apply(len)
        df = df.sort_values(by='user_id')
        df.to_csv(
            Path(next_ckpt_dir) / 'eval_result.csv',
            columns=['user_id',
                     'clicked_news',
                     'candidate_news',
                     'num_clicked_news',
                     'predictions'],
            index=False
        )
        print(compute_metrics((predictions, labels)))
    else:
        raise ValueError('')

# TODO dropout?
# TODO drop_insuffcient?
# TODO bert attention layers?
if __name__ == '__main__': 
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    args = parse_args()
    fix_all_seeds(seed=args.seed)
    # data_preprocessing(args, 'train')
    # data_preprocessing(args, 'valid')
    args.max_dataset_size = 1000
    args.epochs = 1
    args.mode = 'valid'
    main(args)
