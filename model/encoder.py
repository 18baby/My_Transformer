from torch import nn

from blocks.encoder_layer import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 임배딩 생성
        self.emb = TransformerEmbedding(d_model=d_model,               # 입력 차원
                                        max_len=max_len,               # 최대 길이
                                        vocab_size=enc_voc_size,       # 입력 단어 크기
                                        drop_prob=drop_prob,
                                        device=device)
        
        # 인코더 n_layer 만큼 쌓기
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                    for _ in range(n_layers)])
    
    def forward(self, x, src_mask):
        x = self.emb(x)    # 임배딩 진행

        for layer in self.layers:
            x = layer(x, src_mask)

        return x