from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_name, latest_weights_file_path
import matplotlib.pyplot as plt
import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import json
import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
# from validation_ import run_validation

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

# Initialize lists to store metrics
train_losses = []
val_losses = []
# bleu = []
cer = []
wer = []


def greedy_batch_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    batch_size = source.size(0)
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    all_probs = torch.zeros(batch_size, max_len, tokenizer_tgt.get_vocab_size(), device=device)
    completed = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i in range(1, max_len):
        if completed.all():
            break
        decoder_mask = causal_mask(i).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        prob = model.project(out[:, -1])
        next_word = prob.argmax(dim=-1)
        
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)
        completed |= (next_word == eos_idx)
        
        # Fill probability tensor only for active sequences
        active = ~completed
        all_probs[active, i] = prob[active]

    return all_probs

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device):
    model.eval()
    total_loss = 0
    metric_cer = torchmetrics.CharErrorRate().to(device)
    metric_wer = torchmetrics.WordErrorRate().to(device)
    # metric_bleu = torchmetrics.BLEUScore().to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc="Validation Batches", leave=False)
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            target = batch["label"].to(device)

            model_out_probs = greedy_batch_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            model_out_ids = model_out_probs.argmax(dim=-1)

            for idx in range(encoder_input.size(0)):
                source_text = tokenizer_src.decode(encoder_input[idx].cpu().tolist())
                target_text = tokenizer_tgt.decode(target[idx].cpu().tolist())
                predicted_text = tokenizer_tgt.decode(model_out_ids[idx].cpu().tolist())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(predicted_text)

            model_out_probs = model_out_probs.contiguous().view(-1, tokenizer_tgt.get_vocab_size())

            # print("model_out_probs shape:",model_out_probs.size())

            target = target.contiguous().view(-1)

            # print("target shape:",target.size())

            loss = loss_fn(model_out_probs, target)
            total_loss += loss.item() * encoder_input.size(0)
            
            # Safe loggin of the loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

    avg_loss = total_loss / len(validation_ds.dataset)
    cer_data = metric_cer(predicted, expected).item()
    wer_data = metric_wer(predicted, expected).item()
    # bleu_data = metric_bleu(predicted, expected).item()

    return avg_loss, cer_data, wer_data



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

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
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    directory_path = Path('/content/drive/MyDrive/Models/pytorch-transformer/weights')
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    print(preload)
    model_filename = latest_weights_file_path(directory_path,config=config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        print("initial_epoch:", initial_epoch,"global_step:", global_step)
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, initial_epoch + config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        total_train_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            total_train_loss = total_train_loss + loss
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss.item())
        print(f"Epoch {epoch:02d} - Average Train Loss: {avg_train_loss:.4f}")

        val_loss,cer_data,wer_data = run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        print(f"Epoch {epoch:02d} - Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)
        cer.append(cer_data)
        wer.append(wer_data)
        # bleu.append(bleu_data)


        # Save the model at the end of every epoch
        model_filepath = directory_path / (config['model_basename'] + str(epoch) + '.pt')
        print(f"Model weights will be saved at: {model_filepath}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filepath)
    
    save_plot(train_losses, val_losses,directory_path = Path('/content/drive/MyDrive/Models/pytorch-transformer/weights'))
    save_json(directory_path = Path('/content/drive/MyDrive/Models/pytorch-transformer/weights'))


def save_plot(train_losses, val_losses, directory_path):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(cer, label='CER')
    plt.title('Character Error Rate (CER) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('CER')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(wer, label='WER')
    plt.title('Word Error Rate (WER) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.legend()
    
    # plt.subplot(2, 2, 4)
    # plt.plot(bleu, label='BLEU Score')
    # plt.title('BLEU Score over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('BLEU Score')
    # plt.legend()

    plt.tight_layout()
    plt.savefig(directory_path / 'training_validation_metrics.png')
    plt.close()

def save_json(directory_path):
    data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'cer': cer,
        'wer': wer
        # 'blue': bleu
    }

    with open(directory_path / 'metrics.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    