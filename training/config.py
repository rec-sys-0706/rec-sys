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

    # Model training
    tf_threshold = 1             # Term frequencies threshold.
    num_tokens_title = 20        # The number of tokens in title. (context_length)
    num_tokens_abstract = 50     # The number of tokens in abstract.
    num_clicked_news_a_user = 50 # The number of clicked news sampled for each user.
    vocab_size = 68878 + 1
    embedding_dim = 300
    num_heads = 6
    negative_sampling_ratio = 3
    max_epochs = 20
    valid_interval = 1000         # The interval for validation checks between steps.
    patience = 3                  # Early stopping.
    learning_rate = 0.001
    train_batch_size = 32
    valid_batch_size = 128
    use_pretrained_embedding = True







    def __init__(self):
        pass

# d_embed: 300 initialized by the Glove
# n_heads: 16
# dropout_rate: 20%
# batch_size: 64
# train/valid: 10%
# TODO pydantic
