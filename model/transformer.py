import torch
from torch import nn 

from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, 
                 dec_voc_size, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob,  device):
        """
        * src_pad_idx, trg_pad_idx: 임베딩 차원을 맞추기 위한 padding 토큰
        * trg_sos_idx : trg 시작 토큰 인덱스
        """
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size = enc_voc_size, 
                            max_len = max_len,
                            d_model = d_model, 
                            ffn_hidden = ffn_hidden,
                            n_head = n_head, 
                            n_layers = n_layers, 
                            drop_prob = drop_prob, 
                            device=device)
        
        self.decoder = Decoder(dec_voc_size = dec_voc_size, 
                            max_len = max_len,
                            d_model = d_model, 
                            ffn_hidden = ffn_hidden,
                            n_head = n_head, 
                            n_layers = n_layers, 
                            drop_prob = drop_prob, 
                            device=device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)   # padding mask
        trg_mask = self.make_trg_mask(trg)   # look ahead mask
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)        
        """
        padding mask
        src = [batch_size, seq_len]
        mask -> [batch_size, 1, 1, seq_len]
        """
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        """
        look_ahead_mask 
        trg = [batch_size, trg_len]
        mask -> [batch_size, 1, seq_len, 1]
        """
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)  # torch.tril: 하삼각행렬 생성
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask  # [batch_size, 1, trg_len, trg_len]
