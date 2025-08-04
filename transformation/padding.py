""" Class Pre
Purpose: Creates a PyTorch dataset for translation tasks. 
It tokenizes sentence pairs, adds <SOS>, <EOS>, <PAD> tokens, and generates input/output tensors and masks for Transformer models.

Inputs:
ds	                - Dataset (dicts)	Dataset containing translation pairs (e.g., from HuggingFace load_dataset).
tokenizer_src       - Tokenizer	Tokenizer for the source language.
tokenizer_tgt       - Tokenizer	Tokenizer for the target language.
src_lang            - Source language code (e.g., 'en').
tgt_lang            - Target language code (e.g., 'fr').
seq_len	            - Fixed sequence length for input/output tensors.


Output of __len__()
The number of translation pairs (samples) in the dataset.


Outputs from __getitem__ (returned by dataset[idx]:
encoder_input	   - Tensor(seq_len)	Tokenized source sentence with <SOS>, <EOS>, and <PAD> tokens.
decoder_input	   - Tensor(seq_len)	Tokenized target sentence with <SOS> and padding.
encoder_mask	   - Tensor(1,1,L)	Boolean mask (1 for real tokens, 0 for padding) for the encoder.
decoder_mask	   - Tensor(1,L,L)	Causal mask + padding mask for the decoder.
label	           - Tensor(seq_len)	Target sentence tokens with <EOS> and padding, used for loss computation.
src_text	       - Original source sentence text.
tgt_text	       - Original target sentence text.

"""



import torch
import torch.nn as nn
from torch.utils.data import Dataset



class Padding(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1,seq_len) & (1,seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


