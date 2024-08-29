import torch
class BaseConfig():
    # System
    model_name = 'NRMS'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = './data/train'
    val_dir = './data/valid'
    test_dir = './data/test'
    ckpt_dir = './checkpoint'
    glove_embedding_path = './data/glove.840B.300d/glove.840B.300d.txt'
    num_workers = 1

    # Model
    tf_threshold = 1             # Term frequencies threshold.
    negative_sampling_ratio = 3
    num_tokens_title = 20        # The number of tokens in title. (context_length)
    num_tokens_abstract = 50     # The number of tokens in abstract.
    num_clicked_news_a_user = 50 # The number of clicked news sampled for each user.
    vocab_size = 68878 + 1
    embedding_dim = 300
    num_heads = 6
    dropout_rate = 0.2
    # Training Process
    max_epochs = 10
    valid_interval = 1000         # The interval for validation checks between steps.
    patience = 3                  # Early stopping.
    learning_rate = 0.0005
    train_batch_size = 64
    valid_batch_size = 512
    use_pretrained_embedding = False # True



    # In original NRMS paper.
    # d_embed: 300 initialized by the Glove
    # n_heads: 16
    # d_query: 16 # !embedding_dim // n_heads in my case
    # dim of additive attention is 200
    # negative_sampling_ratio: 4
    # dropout_rate: 20% to word embedding
    # batch_size: 64
    # train/valid: 10%


    def __init__(self):
        pass

# TODO pydantic
