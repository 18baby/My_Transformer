
# 디코더 block 생성

from torch import nn 

from layers.norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedFoward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # Masked Multihead attention
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        # Multihead attention
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # FFNN
        self.ffnn = PositionWiseFeedFoward(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        
        # ===== Layer 1 =====
        # masked multihead attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # dropout, add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # ===== Layer 2 =====
        if enc is not None:
            # mulihead attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # dropout, add & norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # ===== Layer 3 =====
        _x = x
        # ffnn
        x = self.ffnn(x)

        # dropout, add & norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x