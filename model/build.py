from .model import build_transformer

def get_model(config, 
              vocab_src_len, 
              vocab_tgt_len):
    
    model = build_transformer(vocab_src_len, 
                              vocab_tgt_len, 
                              config["seq_len"], 
                              config['seq_len'], 
                              d_model=config['d_model'])
    
    return model