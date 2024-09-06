import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import get_config, get_weights_file_path

from pathlib import Path

from dataset import BilingualDataset, attention_mask
from model import build_transformer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def create_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=[
            '[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),
                                      trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def get_ds(config):
    raw_ds = load_dataset(
        'opus_books', f"{config['lang_src']}-{config['lang_tgt']}",
        split='train')
    
    src_tokenizer = create_tokenizer(config, raw_ds, config['lang_src'])
    tgt_tokenizer = create_tokenizer(config, raw_ds, config['lang_tgt'])

    train_ds_size = int(len(raw_ds) * 0.9)
    val_ds_size = len(raw_ds) - train_ds_size
    raw_train_ds, raw_val_ds = random_split(raw_ds, (train_ds_size,
                                                     val_ds_size))
    
    train_ds = BilingualDataset(raw_train_ds, src_tokenizer, tgt_tokenizer,
                                config['lang_src'], config['lang_tgt'],
                                config['seq_len'])
    
    val_ds = BilingualDataset(raw_val_ds, src_tokenizer, tgt_tokenizer,
                              config['lang_src'], config['lang_tgt'],
                              config['seq_len'])
    
    max_tgt_len = -1
    max_src_len = -1

    for item in raw_ds:
        src_enc = src_tokenizer.encode(
            item['translation'][config['lang_src']]).ids
        tgt_enc = tgt_tokenizer.encode(
            item['translation'][config['lang_tgt']]).ids
        max_src_len = max(max_src_len, len(src_enc))
        max_tgt_len = max(max_tgt_len, len(tgt_enc))

    print(f"Max source sentence len: {max_src_len}")
    print(f"Max target sentence len: {max_tgt_len}")

    train_dl = DataLoader(train_ds, batch_size=config['batch_size'],
                          shuffle=True)
    val_dl = DataLoader(val_ds, 1, shuffle=True)

    return train_dl, val_dl, src_tokenizer, tgt_tokenizer


def get_model(config, src_vocab_sz, tgt_vocab_sz):
    return build_transformer(src_vocab_sz, tgt_vocab_sz, config['seq_len'],
                             config['seq_len'], config['d_model'])


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, src_tokenizer, tgt_tokenizer = get_ds(config)

    model = get_model(config, src_tokenizer.get_vocab_size(),
                      tgt_tokenizer.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config['experiment_name'])

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)


    init_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print("Preloading model: " + model_filename)
        state = torch.load(model_filename)
        optim.load_state_dict(state['optimizer_state_dict'])
        init_epoch = state['epoch'] + 1
        global_step = state['global_step']

    epochs = int(config['num_epochs'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(init_epoch, epochs):
        model.train()
        loop = tqdm(train_dl, desc=f"Training epoch {epoch:02d}")
        
        for batch in loop:
            # (B, seq_len) 
            encoder_input = batch['encoder_input'].to(device)
            # (B, seq_len) 
            decoder_input = batch['decoder_input'].to(device)
            # (B, 1, 1, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)
            # (B, 1, seq_len, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)

            # (B, seq_len, d_model)
            encoder_out = model.encode(encoder_input, encoder_mask)
            # (B, seq_len, d_model)
            decoder_out = model.decode(encoder_out, encoder_mask,
                                       decoder_input, decoder_mask)
            # (B, seq_len, tgt_dict_size)
            projection = model.project(decoder_out)

            label = batch['label'].to(device)
            # (B, seq_len, tgt_dict_size) > (B * seq_len, tgt_dict_size)
            loss_val = loss_fn(projection.view(
                -1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            loop.set_postfix({"loss": f"{loss_val.item():6.3f}"})

            writer.add_scalar("train loss", loss_val.item(), global_step)
            writer.flush()

            loss_val.backward()

            optim.step()
            optim.zero_grad()

            global_step += 1
            
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            'global_step': global_step
        })

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    model = train_model(config)
