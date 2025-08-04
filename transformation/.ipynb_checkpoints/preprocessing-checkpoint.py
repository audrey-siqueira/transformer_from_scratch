from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from .tokenization import get_or_build_tokenizer
from .padding import Padding



def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    #Create tokenizers 
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #Dataset filtering by size
    ds_raw = [
        item for item in ds_raw
        if len(tokenizer_src.encode(item['translation'][config['lang_src']]).ids) <= config['seq_len'] - 2
        and len(tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids) <= config['seq_len'] - 1
    ]

    #Splitting number of samples for training and validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    #Preprocessing train data
    train_ds = Padding(train_ds_raw, 
                   tokenizer_src, 
                   tokenizer_tgt, 
                   config['lang_src'], 
                   config['lang_tgt'], 
                   config['seq_len'])
    
    #Batch & Shuffle train data
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)


    #Preprocessing validation data
    val_ds = Padding(val_ds_raw, 
                 tokenizer_src, 
                 tokenizer_tgt, 
                 config['lang_src'], 
                 config['lang_tgt'], 
                 config['seq_len'])

    #Batch & Shuffle validation data
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
