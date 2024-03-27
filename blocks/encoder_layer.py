
# 인코더 block 생성

from torch import nn 

from layers.norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedFoward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # Multihead attention
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffnn = PositionWiseFeedFoward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        
        # ===== Layer 1 =====
        # mulihead attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # dropout
        x = self.dropout1(x)

        # add & norm
        x = self.norm1(x + _x)

        # ===== Layer 2 =====
        # ffnn
        x = self.ffnn(x)

        # dropout
        x = self.dropout2(x)

        # add & norm
        x = self.norm2(x + _x)

        return x