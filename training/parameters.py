from dataclasses import dataclass
import argparse
from typing import Any, Literal

import torch

@dataclass
class Arguments:
    # Directory
    train_dir: str
    val_dir: str
    ckpt_dir: str | None
    glove_embedding_path: Literal['data\glove.6B\glove.6B.300d.txt', 'data/glove.840B.300d/glove.840B.300d.txt']
    # Model
    max_vocab_size: int
    min_frequency: int
    negative_sampling_ratio: int
    num_tokens_title: int
    num_tokens_abstract: int
    max_clicked_news: int
    embedding_dim: int
    num_heads: int
    dropout_rate: float
    max_dataset_size: int
    use_category: bool
    use_full_candidate: bool
    # Training Process
    epochs: int
    eval_strategy: Literal['steps', 'epoch']
    eval_steps: int
    patience: int
    early_stop: bool
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    drop_insufficient: bool
    metric_for_best_model: Literal['loss', 'auc', 'acc']
    greater_is_better: bool
    # System
    mode: Literal['train', 'valid']
    seed: int
    cpu: bool
    model_name: Literal['NRMS', 'NRMS-Glove', 'NRMS-BERT']
    pretrained_model_name: Literal['distilbert-base-uncased', 'bert-base-uncased']
    reprocess_data: bool
    regenerate_dataset: bool 
    generate_bertviz: bool
    generate_tsne: bool
    continue_training: bool
    device: Any = None
    def __post_init__(self):
        self.drop_insufficient = True
        self.cpu = False # utils.parse_argv
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.model_name == 'NRMS-BERT':
            self.embedding_dim = 768
        if not self.embedding_dim % self.num_heads == 0:
            raise ValueError(f'embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})')

def parse_args() -> Arguments:
    parser = argparse.ArgumentParser()
    # System
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--cpu', action='store_true', help="Use CPU to run the model. If not set, the model will run on GPU by default.")
    parser.add_argument('--model-name', type=str, default='NRMS')
    parser.add_argument('--pretrained-model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--reprocess-data', action='store_true', help="Whether to redo data-preprocessing even if processed data exists.")
    parser.add_argument('--regenerate-dataset', action='store_true', help="Whether to redo dataset even if .pt exists.")
    parser.add_argument('--generate-bertviz', action='store_true', help="Whether to generate BERTViz.")
    parser.add_argument('--generate-tsne', action='store_true', help="Whether to generate t-SNE.")
    parser.add_argument('--continue-training', action='store_true', help="Whether to continue training.")
    # Directory
    parser.add_argument('--train-dir', type=str, default='data/train')
    parser.add_argument('--val-dir', type=str, default='data/valid')
    parser.add_argument('--ckpt-dir', type=str, default=None, help='Specify a name for checkpoint directory, it will get last checkpoint.')
    parser.add_argument('--glove-embedding-path', type=str, default='data\glove.840B.300d\glove.840B.300d.txt')
    # Model
    parser.add_argument('--max-vocab-size', type=int, default=30000, help="The maximum number of unique tokens")
    parser.add_argument('--min-frequency', type=int, default=2, help="Term frequency threshold")
    parser.add_argument('--negative-sampling-ratio', type=int, default=1)
    parser.add_argument('--num-tokens-title', type=int, default=24, help="The number of tokens in title (aka. context_length)")
    parser.add_argument('--num-tokens-abstract', type=int, default=50, help="The number of tokens in abstract")
    parser.add_argument('--max-clicked-news', type=int, default=64, help="The maximum number of clicked news sampled for each user")
    parser.add_argument('--embedding-dim', type=int, default=300, help="If using `NRMS-BERT`, then the embedding_dim will be 768")
    parser.add_argument('--num-heads', type=int, default=6, help="The number of attention heads")
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--max-dataset-size', type=int, default=60000, help="The upper limit of the dataset size.")
    parser.add_argument('--use-category', action='store_true', help="Use category as features.")
    parser.add_argument('--use-full-candidate', action='store_true', help="Use full candidate set.")
    # Training Process
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval-strategy', type=str, default='epoch', help="The timing to evaluate model, it could be either `steps` or `epoch`")
    parser.add_argument('--eval-steps', type=int, default=1000, help="The interval for evaluation between steps")
    parser.add_argument('--early-stop', type=bool, default=False, help="Whether to use early stopping")
    parser.add_argument('--patience', type=int, default=3, help="Patience early stopping")
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--eval-batch-size', type=int, default=512)
    parser.add_argument('--drop-insufficient', action='store_true', help="Drop row which is insufficient in clicked_news, clicked_candidate")
    parser.add_argument('--metric-for-best-model', type=str, default='loss', help="Which metric should be monitored while early stopping")
    parser.add_argument('--greater-is-better', type=bool, default=False, help="Whether metric greater is better")

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