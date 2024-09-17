from dataclasses import dataclass
import argparse
from typing import Any

import torch

@dataclass
class Arguments:
    # System
    seed: int
    cpu: bool
    device: Any
    # Directory
    train_dir: str
    val_dir: str
    test_dir: str
    ckpt_dir: str
        # glove_embedding_path: str
    # Model
    tf_threshold: int
    negative_sampling_ratio: int
    num_tokens_title: int
    num_tokens_abstract: int
    num_clicked_news: int
        # vocab_size: int
    embedding_dim: int
    num_heads: int
    dropout_rate: float

    # Training Process
    epochs: int
    valid_interval: int
    patience: int
    learning_rate: float
    train_batch_size: int
    valid_batch_size: int
    drop_insufficient: bool

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
    parser.add_argument('--tf-threshold', type=int, default=1, help="Term frequencies threshold")
    parser.add_argument('--negative-sampling-ratio', type=int, default=1)
    parser.add_argument('--num-tokens-title', type=int, default=24, help="The number of tokens in title (aka. context_length)")
    parser.add_argument('--num-tokens-abstract', type=int, default=50, help="The number of tokens in abstract")
    parser.add_argument('--num-clicked-news', type=int, default=64, help="The number of clicked news sampled for each user")
    # vocab_size = 68878 + 1
    parser.add_argument('--embedding-dim', type=int, default=768)
    parser.add_argument('--num-heads', type=int, default=6, help="The number of attention heads")
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    # Training Process
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--valid-interval', type=int, default=1000, help="The interval for validation checks between steps")
    parser.add_argument('--patience', type=int, default=3, help="Patience early stopping")
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--valid-batch-size', type=int, default=512)
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