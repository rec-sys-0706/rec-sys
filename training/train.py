import logging
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from safetensors.torch import load_file

from model.NRMS import NRMS, NRMS_BERT
from parameters import Arguments, parse_args
from utils import CustomTokenizer, time_since, get_datetime_now, fix_all_seeds
from dataset import NewsDataset
from evaluate import nDCG, ROC_AUC, recall, accuracy
evaluate = lambda _pred, _true: (recall(_pred, _true), ROC_AUC(_pred, _true), nDCG(_pred, _true, 5), nDCG(_pred, _true, 10), accuracy(_pred, _true))


import pdb
def compute_metrics(eval_preds):
    predictions, label_ids = eval_preds
    pdb.set_trace()
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

def get_log_dirs(args: Arguments):
    """Get log_dir and ckpt_dir by using Datatime"""
    DATETIME_NOW = get_datetime_now()

    log_dir = f'./runs/{DATETIME_NOW}'
    if not Path('runs').exists():
        Path('runs').mkdir()
    Path(log_dir).mkdir()

    with open(f'{log_dir}/args.json', 'w') as f:
        args_dict = {k: str(v) for k, v in vars(args).items()}
        json.dump(args_dict, f)

    ckpt_dir = f'./checkpoints/{DATETIME_NOW}'
    if not Path('checkpoints').exists():
        Path('checkpoints').mkdir()
    Path(ckpt_dir).mkdir()

    return log_dir, ckpt_dir

def get_trainer_args(args: Arguments, ckpt_dir, log_dir) -> TrainingArguments:
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
        logging_dir=log_dir, # TensorBoard
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        seed=args.seed
        # logging_steps=
        #logging_strategy=
        # label_names=[]
        # optim=
        # warmup_steps=500
        # weight_decay=0.01
        # dataloader_drop_last=
        # push_to_hub=True,
    )  

def train(args: Arguments):
    fix_all_seeds(args.seed)
    log_dir, ckpt_dir = get_log_dirs(args)
    # Tokenizer & Model
    tokenizer = CustomTokenizer(args)
    model = get_model(args, tokenizer)
    # Prepare Datasets
    logging.info(f'Loading datasets...')
    start_time = time.time()
    train_dataset = NewsDataset(args, tokenizer, mode='train')
    valid_dataset = NewsDataset(args, tokenizer, mode='valid')
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')



    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience,
    )
    training_args = get_trainer_args(args, ckpt_dir, log_dir)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_collate,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # callbacks=[early_stopping_callback]
    )
    trainer.train()

def valid(args: Arguments):
    fix_all_seeds(args.seed)
    log_dir, ckpt_dir = get_log_dirs(args)
    # Tokenizer & Model
    tokenizer = CustomTokenizer(args)
    model = get_model(args, tokenizer)
    # load model
    last_checkpoint = get_last_checkpoint('checkpoints/2024-09-25T231941')
    tensors = load_file(last_checkpoint + "/model.safetensors")
    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if key in tensors:
            model_state_dict[key] = tensors[key]
    model.load_state_dict(model_state_dict)
    # Prepare Datasets
    logging.info(f'Loading datasets...')
    start_time = time.time()
    valid_dataset = NewsDataset(args, tokenizer, mode='valid')
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')
    validation_args = get_trainer_args(args, ckpt_dir, log_dir)
    trainer = Trainer(
        model,
        validation_args,
        eval_dataset=valid_dataset,
        data_collator=default_collate,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.predict(valid_dataset)
if __name__ == '__main__': 
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    args = parse_args()
    valid(args)

# TODO valid
# TODO test
# TODO dropout?