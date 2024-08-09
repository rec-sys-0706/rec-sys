import torch
class BaseConfig():
    model_name = 'NRMS'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dir = './data/train'
    val_dir = './data/valid'
    test_dir = './data/test'
    
    
    tf_threshold = 1             # term frequencies threshold.
    num_tokens_title = 20        # The number of tokens in title. (context_length)
    num_tokens_abstract = 50     # The number of tokens in abstract.
    num_clicked_news_a_user = 50 # The number of clicked news sampled for each user.
    vocab_size = 68878 + 1
    embedding_dim = 30
    num_heads = 6
    negative_sampling_ratio=3
    max_epochs = 2
    valid_interval = 100         # The interval for validation checks between steps.
    train_batch_size = 32
    valid_batch_size = 32








    def __init__(self):
        pass

# d_embed: 300 initialized by the Glove
# n_heads: 16
# dropout_rate: 20%
# batch_size: 64
# train/valid: 10%
# TODO pydantic
