import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import get_config, get_weights_file_path, latest_weights_file_path
from model.build import get_model
from transformation.preprocessing import get_ds
from inference.validation import run_validation


def train_model(config):
    
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
        
    elif (device == 'mps'):
        print(f"Device name: <mps>")
        
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
        
    device = torch.device(device)




    #Make Weights folder
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    #load Dataset, Tokenizer generation, filter Dataset by lenght, padding
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    
    #Build Model
    model = get_model(config, 
                      tokenizer_src.get_vocab_size(), 
                      tokenizer_tgt.get_vocab_size()).to(device)
    
    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])


    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    

    #Preload Model
    preload = config['preload']
    if preload == 'latest':
         model_filename = latest_weights_file_path(config)
    elif preload:
         model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    
    #Training
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    #Iteration over epochs
    for epoch in range(initial_epoch, config['num_epochs']):
        
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch_idx, batch in enumerate(batch_iterator):

            #Getting data if is last batch from last epoch
            is_last_epoch = (epoch == config['num_epochs'] - 1)
            is_last_batch = (batch_idx == len(train_dataloader) - 1)

            #Saving
            if is_last_epoch and is_last_batch:
                #Activate save debug in every model layer
                for name, module in model.named_modules():
                    if hasattr(module, 'save_debug'):
                        module.save_debug = True
                        print(f'Ativado save_debug para: {name}')


            #Encoding
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask =  batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)

            
            #Decoding
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            decoder_mask =  batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            decoder_output = model.decode(encoder_output, 
                                          encoder_mask, 
                                          decoder_input, 
                                          decoder_mask) # (B, seq_len, d_model)

            

            #Projection Layer
            proj_output    = model.project(decoder_output) # (B, seq_len, vocab_size)

            #Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)
            

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            #Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


            #Deactivate save debug in every model layer
            if is_last_epoch and is_last_batch:
                for name, module in model.named_modules():
                    if hasattr(module, 'save_debug'):
                        module.save_debug = False
                        #print(f'Desativado save_debug para: {name}')


            global_step += 1
            
            

        #Run validation at the end of every epoch
        run_validation(model, 
                       val_dataloader, 
                       tokenizer_src, 
                       tokenizer_tgt, 
                       config['seq_len'], 
                       device, 
                       lambda msg: batch_iterator.write(msg), 
                       global_step, 
                       writer)
        

        #Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({ 'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'global_step': global_step }, 
                      model_filename)
        





if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
