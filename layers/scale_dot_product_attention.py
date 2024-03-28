# attention 진행

import math
from torch import nn 

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        입력 = [batch_size, n_head, n_seq, d_model]
        n_seq: 입력 단어수
        d_model: 단어 임베딩 차원
        """
        batch_size, n_head, n_seq, d_model = k.size()

        # 1. attention score 계산
        k_t = k.transpose(2, 3)   
        attention_score = (k_t @ q)/math.sqrt(d_model)

        # 2. masking 적용
        if mask is not None:
            score = score.masked_fill(mask==0, -1e9)     # 디코더에서 현시점 이후의 단어들을 모두 -10000으로 마스킹 진행

        # 3. attention distribution 계산
        attention_dist = self.softmax(attention_score)

        # 4. attention value 계산
        v = attention_dist @ v

        return v, attention_score
