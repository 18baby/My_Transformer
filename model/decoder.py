from torch import nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 임배딩 생성
        self.emb = TransformerEmbedding(d_model=d_model,               # 입력 차원
                                        max_len=max_len,               # 문장내 최대 단어 개수
                                        vocab_size=dec_voc_size,       
                                        drop_prob=drop_prob,
                                        device=device)
        
        # 인코더 n_layer 만큼 쌓기
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                    for _ in range(n_layers)])
        # 최종 결과 출력
        self.linear = nn.Linear(d_model, dec_voc_size)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)    # 디코더 입력 임배딩 진행

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        out = self.linear(trg)

        return out