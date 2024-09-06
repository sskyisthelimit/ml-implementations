import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang,
                 tgt_lang, seq_len):
        super(BilingualDataset, self).__init__()

        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len


        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_src_sentence = self.src_tokenizer.encode(src_text).ids
        enc_tgt_sentence = self.tgt_tokenizer.encode(tgt_text).ids

        src_padd_len = self.seq_len - len(enc_src_sentence) - 2
        tgt_padd_len = self.seq_len - len(enc_tgt_sentence) - 1

        if src_padd_len < 0:
            raise ValueError("Source sentence can't be longer that seq_len!")
        elif tgt_padd_len < 0:
            raise ValueError("Target sentence can't be longer that seq_len!")            

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_src_sentence, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * src_padd_len,
                             dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_tgt_sentence, dtype=torch.int64),
                torch.tensor([self.pad_token] * tgt_padd_len,
                             dtype=torch.int64)
            ]
        )
  
        label_sentence = torch.cat(
            [
                torch.tensor(enc_tgt_sentence, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * tgt_padd_len,
                             dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label_sentence.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            # (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token
                             ).unsqueeze(0).unsqueeze(0).int(),
            # (1, 1, seq_len) & (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token
                             ).unsqueeze(0).unsqueeze(0).int(
                             ) & attention_mask(self.seq_len),
            "label": label_sentence,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def attention_mask(seq_len):
    triu_mask = torch.triu(input=torch.ones(1, seq_len, seq_len),
                           diagonal=1).type(torch.int)
    return triu_mask == 0