import string
import time
import random
from datetime import datetime
from typing import Literal
from pathlib import Path
import logging

import numpy as np
import torch
import pandas as pd
import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerFast, BertTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from parameters import Arguments, parse_args
from pydantic import BaseModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from tqdm import tqdm
class Encoding(BaseModel):
    input_ids: list[int]
    token_type_ids: list[int]
    attention_mask: list[int]

class GroupedNews(BaseModel):
    title: list[Encoding]
    abstract: list[Encoding]

class Example(BaseModel):
    clicked_news: GroupedNews
    candidate_news: GroupedNews
    clicked: list[int]
REMAINS_CATEGORY = ['autos', 'entertainment', 'finance', 'foodanddrink', 'health', 'weather', 'middleeast', 'sports', 'travel', 'tv']
NOT_IN_CNN = ['movies', 'music', 'video'] # 'games', 'kids', 

class CustomTokenizer:
    """CustomTokenizer, wrapping HuggingFace Tokenizer inside"""
    def __init__(self, args: Arguments):
        self.SPECIAL_TOKENS = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        self.args = args
        self.tokenizer_file = Path(args.train_dir) / 'tokenizer.json'
        self.categorizer_file = Path(args.train_dir) / 'categorizer.json'

        self.__categorizer = self.__build_categorizer()
        if args.model_name == 'NRMS' or args.model_name == 'NRMS-Glove':
            if not self.tokenizer_file.exists():
                logging.info("Tokenizer file not detected.")
                logging.info("Start building tokenizer and saving to `train_dir`.")
                self.__tokenizer = self.__build_tokenizer()
            else:
                self.__tokenizer = self.__load_tokenizer()
            self.__tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        elif args.model_name == 'NRMS-BERT':
            self.__tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        else:
            pass
            # self.ENC = tiktoken.get_encoding("o200k_base") # gpt-4o

        self.title_padding = self.encode_title('')
        self.abstract_padding = self.encode_abstract('')
        self.vocab_size = self.__tokenizer.vocab_size
        self.num_category = self.__categorizer.vocab_size

    def __call__(self, *args, **kwargs):
        return self.__tokenizer(*args, **kwargs)

    def tokenize(self, text: str) -> list[str]:
        # return [ENC.decode([token]) for token in tokens] # TODO optmize
        # return list(text.lower()) # character only, 123
        # return re.sub(r'[^a-z0-9\s]', '', text.lower()).split() # 36306
        # return nltk.word_tokenize(text.lower()) # 37539
        pass
        # For encoding: [int(word2int.get(token, 0)) for token in tokenize(text)]

    def encode(self):
        pass
    def encode_title(self, text) -> Encoding:
        return self.__tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_title)
    def encode_abstract(self, text) -> Encoding:
        return self.__tokenizer(text, padding='max_length', truncation=True, max_length=self.args.num_tokens_abstract)
    def encode_category(self, category):
        return self.__categorizer.vocab.get(category, 0)
    def decode(self):
        pass
        # TODO
    def decode_category(self, category_id):
        return self.__categorizer.decode(category_id)
    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.__tokenizer.convert_ids_to_tokens(*args, **kwargs)
    def __build_tokenizer(self) -> PreTrainedTokenizerFast:
        args = self.args
        news = pd.read_csv(Path(args.train_dir) / 'news.tsv',
                        sep='\t',
                        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                        index_col='news_id')
        news['abstract'] = news['abstract'].fillna('')
        texts = pd.concat((news['title'], news['abstract'])).tolist()

        tokenizer = Tokenizer(models.WordLevel(unk_token=self.SPECIAL_TOKENS['unk_token']))
        tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFC(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=args.max_vocab_size,
                                            min_frequency=args.min_frequency,
                                            special_tokens=list(self.SPECIAL_TOKENS.values()))

        tokenizer.train_from_iterator(texts, trainer)
        tokenizer.save(self.tokenizer_file.__str__())
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        return tokenizer

    def __load_tokenizer(self) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_file.__str__())
        return tokenizer

    def __build_categorizer(self) -> PreTrainedTokenizerFast:
        logging.info("Building categorizer...")
        args = self.args
        news = pd.read_csv(Path(args.train_dir) / 'news.tsv',
                        sep='\t',
                        names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'],
                        index_col='news_id')
        news['category'] = news.apply(reclassify_category, axis=1)
        # categories = pd.concat([news['category'], news['subcategory']]).unique().tolist() # TODO delete
        categories = news['category'].dropna().sort_values().unique().tolist()
        vocab = {category: idx for idx, category in enumerate(categories, start=1)}
        vocab.update({'<unk>': 0})
        _categorizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        _categorizer.pre_tokenizer = pre_tokenizers.Whitespace()
        _categorizer.save(self.categorizer_file.__str__())
        categorizer = PreTrainedTokenizerFast(tokenizer_object=_categorizer, unk_token="<unk>")
        return categorizer
    def save_pretrained(self, *args, **kwargs):
        self.__tokenizer.save_pretrained(*args, **kwargs)
    def get_vocab(self) -> dict[str, int]:
        return self.__tokenizer.get_vocab()


def reclassify_category(row):
    if row['category'] == 'finance':
        return 'economy-and-finance'
    if row['category'] == 'middleeast':
        return 'area-world'
    if row['category'] == 'foodanddrink':
        return 'food-and-drink'
    if row['category'] in REMAINS_CATEGORY:
        return row['category']

    if row['category'] in NOT_IN_CNN:
        return row['category']

    if row['category'] in ['news', 'lifestyle']:
        c = row['category'] + ' ' + row['subcategory']
        if c in ['news causes', 'news causes-military-appreciation', 'news causes-poverty', 'news newscrime', 'lifestyle causes-green-living']:
            return 'social-issues' # 社會議題
        elif c in ['news causes-disaster-relief', 'news causes-environment']:
            return 'weather' # 環境、氣候、天氣
        elif c in ['news elections-2020-us', 'news newselection2020', 'news indepth', 'news newspolitics', 'news newsworldpolitics']:
            return 'politics' # 選舉與政治
        elif c in ['news factcheck', 'news newsfactcheck', 'news narendramodi_opinion', 'news newsopinion', 'news newstvmedia']:
            return 'comment' # 評論
        elif c in ['news newsus', 'news newsworld', 'news newsnational']:
            return 'area-world' # area-world
        elif c in ['news newsbusiness']:
            return 'business' # 商業
        elif c in ['news newsscience', 'news newsscienceandtechnology', 'news newstechnology']:
            return 'science-and-technology' # 科學與科技
        elif c in ['news personalfinance', 'news newsrealestate']:
            return 'economy-and-finance'
        elif c in ['news empowering-the-planet', 'news newsweather']:
            return 'weather' # 環境、氣候、天氣
        elif c in ['news newsvideo', 'news newsvideos']:
            return 'video'
        elif c in ['lifestyle lifestylefamily', 'lifestyle lifestylefamilyandrelationships', 'lifestyle lifestyleparenting', 'lifestyle lifestylelovesex', 'lifestyle lifestylemarriage', 'lifestyle pregnancyparenting', 'lifestyle advice']:
            return 'emotional' # 家庭、情感相關
        elif c in ['lifestyle lifestylepets', 'lifestyle lifestylepetsanimals', 'lifestyle causes-animals', 'lifestyle lifestyleanimals']:
            return 'pet-and-animal'
        elif c in ['lifestyle lifestylefashion', 'lifestyle lifestylebeauty', 'lifestyle awardstyle', 'lifestyle lifestylecelebstyle']:
            return 'fashion'
        elif c in ['lifestyle lifestyleshopping', 'lifestyle shop-all', 'lifestyle shop-apparel', 'lifestyle shop-books-movies-tv', 'lifestyle shop-computers-electronics', 'lifestyle shop-holidays', 'lifestyle shop-home-goods', 'lifestyle lifestyleshoppinghomegarden']:
            return 'shopping' # 購物
        elif c in ['lifestyle lifestylecleaningandorganizing', 'lifestyle lifestyledecor', 'lifestyle lifestylehomeandgarden']:
            return 'home' # 居家
        elif c in ['lifestyle holidays', 'lifestyle lifestylestyle', 'lifestyle lifestyletravel', 'lifestyle travel', 'lifestyle lifestyle-wedding', 'lifestyle lifestyleweddings']:
            return 'festival'
        elif c in ['lifestyle lifestylecareer']:
            return 'workplace' # 職場
        elif c in ['lifestyle lifestylehoroscope', 'lifestyle lifestylehoroscopefish']:
            return 'astrology' # 占星
    return None  # Return None if no match is found

def draw_tsne(df: pd.DataFrame, tokenizer: CustomTokenizer, random_state: int=42, perplexity: int=30, learning_rate='auto', max_iter=1000):
    distinct_colors = [
        '#d62728',
        '#ff7f0e',
        '#ffed6f',
        '#33a02c',
        '#1f78b4',
        '#17becf',
        '#9467bd',
        '#e377c2',
        '#7f7f7f',
        '#b15928',
    ]
    start_time = time.time()
    # Drop <unk>

    df = df[df['category'].apply(tokenizer.decode_category).isin([
        'autos',
        'economy-and-finance',
        'food-and-drink',
        'health',
        'politics',
        'science-and-technology',
        'social-issues',
        'sports',
        'tv',
        'weather',
        # '<unk>',
        # 'area-world',
        # 'video',
        # 'science-and-technology',
        # 'travel'
    ])]
    # Filter top 10 frequent categories
    filter = df.groupby('category').size().reset_index(name='count').sort_values(by='count', ascending=False).head(10)['category'].unique()
    df = df[df['category'].isin(filter)]

    # Add proportions (if unequal proportions are needed, specify them here)
    proportions = [0.1] * 10  # Equal proportion, adjust if needed

    # Calculate sample sizes for each category
    sample_sizes = [int(p * min(len(df), 10000)) for p in proportions]

    # Sample proportionately
    sampled_data = pd.DataFrame(columns=df.columns)

    for category, size in zip(df['category'].unique(), sample_sizes):
        try:
            sampled_rows = df[df['category'] == category].sample(size)
        except:
            sampled_rows = df[df['category'] == category]
        sampled_data = pd.concat([sampled_data, sampled_rows])

    df = sampled_data.reset_index(drop=True)
    info = df.groupby('category').size().reset_index()
    info['label'] = info['category'].apply(tokenizer.decode_category)
    print(info)
    
    # Step 1: Apply PCA to reduce dimensionality from 768 to 50 (or another suitable value)
    pca = PCA(n_components=50)  # Adjust n_components based on the data structure
    data_pca = pca.fit_transform(df.iloc[:, 2:])  # Assuming df.iloc[:, 2:] has the 768-dim BERT embeddings

    # Step 2: Apply t-SNE to the PCA-reduced data
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter)
    tsne_result = tsne.fit_transform(data_pca)
    df['tsne_x'] = tsne_result[:, 0]
    df['tsne_y'] = tsne_result[:, 1]
    print(f"t-SNE took {time_since(start_time)}")
    # Get unique categories and a continuous colormap
    unique_categories = df['category'].unique()
    # cmap = cm.get_cmap('viridis', len(unique_categories)) # Color too similar

    # Create a color mapping using the continuous colormap
    color_mapping = {category: (f'{tokenizer.decode_category(category)}', distinct_colors[i % len(distinct_colors)])
                     for i, category in enumerate(sorted(unique_categories))}
    # Plot with unique colors
    fig, ax = plt.subplots(figsize=(10, 8))
    for category, (label, color) in color_mapping.items():
        subset = df[df['category'] == category]
        ax.scatter(subset['tsne_x'], subset['tsne_y'], label=category, color=color, edgecolors='black', linewidths=0.5)

    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
                    for label, color in color_mapping.values()]

    ax.legend(handles=legend_elements, title="Categories")

    # Show the plot
    ax.set_xlabel('t-SNE X')
    ax.set_ylabel('t-SNE Y')
    ax.set_title('t-SNE Scatter Plot with Category Labels')
    return fig
    # 1. 去除小樣本
    # 2. id排序
    # 3. 顏色選擇

def draw_tsne_with_ckpt(record_vector_path: str, random_state: int=42, perplexity: int=30, learning_rate='auto', n_iter=1000):
    df = pd.read_csv(record_vector_path)
    args = parse_args()
    tokenizer = CustomTokenizer(args)
    fig = draw_tsne(df, tokenizer, random_state, perplexity, learning_rate, n_iter)
    fig.savefig(f'tsne_{get_datetime_now()}.png')

def time_since(base: float, format: None|Literal['seconds']=None):
    now = time.time()
    elapsed_time = now - base
    if format == 'seconds':
        return elapsed_time
    else:
        return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

def detokenize(word2int_path, seq):
    seq = np.array(seq).astype(str)
    word2int = dict(pd.read_csv(word2int_path,
                                sep='\t',
                                names=['word', 'int'],
                                index_col=False).values)
    int2word = {v: k for k, v in word2int.items()}

    decode_map = np.vectorize(int2word.get)
    return decode_map(seq)

def get_datetime_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%dT%H%M%S")

def format_duration(sec):
    return time.strftime("%H:%M:%S", time.gmtime(sec))

def fix_all_seeds(seed):
    """Fixes RNG seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def tru_pad(tokens: list[str], max_length: int):
    """truncation and padding"""
    if len(tokens) < max_length:
        result = tokens + [0] * (max_length - len(tokens))
    else:
        result = tokens[:max_length]
    attention_mask = [1 if i < len(tokens) else 0 for i in range(max_length)]
    return result, attention_mask
    # ! truncation and padding
    # news[['title', 'title_attention_mask']] = news['title'].apply(lambda t: pd.Series(tru_pad(t, args.num_tokens_title)))
    # news[['abstract', 'abstract_attention_mask']] = news['abstract'].apply(lambda t: pd.Series(tru_pad(t, args.num_tokens_abstract)))

def get_src_dir(args: Arguments, mode) -> Path:
    if mode == 'train':
        src_dir = Path(args.train_dir)
    elif mode == 'valid':
        src_dir = Path(args.val_dir)
    elif mode == 'test':
        src_dir = Path(args.test_dir)
    else:
        raise ValueError(f"[ERROR] Expected `mode` be str['train'|'valid'|'test'] but got `{mode}` instead.")
    return src_dir

def get_suffix(args: Arguments) -> str:
    if args.model_name == 'NRMS-BERT':
        return '_bert'
    else:
        return ''

def parse_argv(argv: list[str]) -> dict:
    argv_dict = {}
    for i in range(1, len(argv), 2):
        if argv[i].startswith('--'):
            k = argv[i]
            v = argv[i+1]
        else:
            k = argv[i]
            v = True
            i -= 1
        argv_dict[k] = v
    return argv_dict

def test_string() -> str:
    mixed_case = "No"
    accents = "HÉLLOcafé"
    unicode = "你好こんにちは😊"
    control = r"\x00\x07"
    currency = "$100 €50 ₹200"
    full_width = "Ｈｅｌｌｏ"
    url = "http://example.com"
    file_path = "C:\\Users\\Name"
    text = (f'   {mixed_case}, {accents}, {unicode}, {string.punctuation}, {control}, {currency}, {full_width}, مرحبا بالعالم , 𝒜𝓁𝓅𝒽𝒶, {url}'
            f', {file_path},½ ⅓ H₂O x² ∑ √ œuvre'
            f'<body></body>   ')
    return text


def list_to_dict(objs: list[dict]):
    """Convert list[dict] to dict[list]"""
    result = {}

    for obj in objs:
        for key, value in obj.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    return result

def dict_to_tensors(obj: dict, dtype=None):
    """Convert dictionary values to tensors recursively"""
    for key, value in obj.items():
        if isinstance(value, dict): # is a dictionary
            dict_to_tensors(value, dtype) # recursively
        else:
            obj[key] = torch.tensor(value, dtype=dtype)
    return obj

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def traverse_object(obj):
    """traverse complex deep structure.
    Usage:
        result = traverse_object([{'a':[{'b':{'c':{'d':[[[['e']]]]}}}]}])
        print(json.dumps(result, ensure_ascii=False, indent=4))
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = traverse_object(value)
        return result
    elif isinstance(obj, list):
        if obj: # if not empty
            return [len(obj), traverse_object(obj[0])] # Expect all element of list are the same, just see the first element.
        else:
            return [0, None]  # return empty list
    elif isinstance(obj, torch.Tensor):
        return f"tensor{tuple(obj.size())}"
    else:
        return obj  # int, str...
