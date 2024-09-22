import logging
import time
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from model.NRMS import NRMS
from parameters import Arguments, parse_args
from utils import CustomTokenizer, time_since, get_datetime_now, fix_all_seeds
from dataset import NewsDataset
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
def train(args: Arguments):
    fix_all_seeds(args.seed)
    if not Path(args.ckpt_dir).exists():
        Path(args.ckpt_dir).mkdir()
    ckpt_now_dir = Path(args.ckpt_dir) / get_datetime_now() 
    ckpt_now_dir.mkdir()

    tokenizer = CustomTokenizer(args)
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

    # Prepare Datasets
    logging.info(f'Loading datasets...')
    start_time = time.time()
    train_dataset = NewsDataset(args, tokenizer, mode='train')
    valid_dataset = NewsDataset(args, tokenizer, mode='valid')
    logging.info(f'Datasets loaded successfully in {time_since(start_time, "seconds"):.2f} seconds.')


    training_args = TrainingArguments(
        ckpt_now_dir,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        save_strategy='epoch',
        num_train_epochs=args.epochs,
        label_names=['clicked'],
        logging_dir=f"./runs/{get_datetime_now()}", # TensorBoard
        load_best_model_at_end=True,
        # logging_steps=
        #logging_strategy=
        # label_names=[]
        # metric_for_best_model='acc'
        # optim=
        # warmup_steps=500
        # weight_decay=0.01
        # dataloader_drop_last=
        # push_to_hub=True,
    ) 
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.patience,
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_collate,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    trainer.train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')
    args = parse_args()
    args.model_name = 'NRMS-Glove'
    # args.glove_embedding_path = 'data/glove.6B/glove.6B.300d.txt'
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
