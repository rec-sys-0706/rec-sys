from dataclasses import dataclass
import argparse
from typing import Any, Literal

import torch

@dataclass
class Arguments:
    # Directory
    train_dir: str
    val_dir: str
    test_dir: str
    ckpt_dir: str
        # glove_embedding_path: str
    # Model
    max_vocab_size: int
    min_frequency: int
    negative_sampling_ratio: int
    num_tokens_title: int
    num_tokens_abstract: int
    num_clicked_news: int
    embedding_dim: int
    num_heads: int
    dropout_rate: float
    tokenizer_mode: Literal['bert', 'gpt-4o', 'vanilla']

    # Training Process
    epochs: int
    eval_strategy: Literal['steps', 'epoch']
    eval_steps: int
    patience: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    drop_insufficient: bool

    # System
    seed: int
    cpu: bool
    device: Any = None
    def __post_init__(self):
        self.drop_insufficient=True
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument('--cpu', action='store_true', help="Use CPU to run the model. If not set, the model will run on GPU by default.")
        # model_name = 'NRMS'
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # num_workers = 1
    parser.add_argument('--seed', type=int, default=1337)
    # Directory
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--val-dir', type=str, default='data/valid')
    parser.add_argument('--test-dir', type=str, default='data/test')
    parser.add_argument('--ckpt-dir', type=str, default='data/checkpoint')
        # glove_embedding_path = './data/glove.840B.300d/glove.840B.300d.txt'
    # Model
    parser.add_argument('--max-vocab-size', type=int, default=30000, help="The maximum number of unique tokens")
    parser.add_argument('--min-frequency', type=int, default=2, help="Term frequency threshold")
    parser.add_argument('--negative-sampling-ratio', type=int, default=1)
    parser.add_argument('--num-tokens-title', type=int, default=24, help="The number of tokens in title (aka. context_length)")
    parser.add_argument('--num-tokens-abstract', type=int, default=50, help="The number of tokens in abstract")
    parser.add_argument('--num-clicked-news', type=int, default=64, help="The number of clicked news sampled for each user")
    parser.add_argument('--embedding-dim', type=int, default=300) # 768
    parser.add_argument('--num-heads', type=int, default=6, help="The number of attention heads")
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--tokenizer-mode', type=str, default='vanilla') # TODO 
    # Training Process
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval-strategy', type=str, default='epoch', help="The timing to evaluate model, it could be either `steps` or `epoch`")
    parser.add_argument('--eval-steps', type=int, default=1000, help="The interval for evaluation between steps")
    parser.add_argument('--patience', type=int, default=3, help="Patience early stopping")
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=512)
    parser.add_argument('--drop-insufficient', action='store_true', help="Drop row which is insufficient in clicked_news, clicked_candidate.")
    # use_pretrained_embedding = False # True

    parsed_args = parser.parse_args()
    args_dict = vars(parsed_args)
    args = Arguments(**args_dict)
    return args
    # In original NRMS paper.
    # d_embed: 300 initialized by the Glove
    # n_heads: 16
    # d_query: 16 # !embedding_dim // n_heads in my case
    # dim of additive attention is 200
    # negative_sampling_ratio: 4
    # dropout_rate: 20% to word embedding
    # batch_size: 64
    # train/valid: 10%


if __name__ == '__main__':
    args = parse_args()
    print(args)