# multi head attention 구현

from torch import nn 
from layers.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    q, k, v = (seq_len, d_model)
    wq, wk, wv = (d_model, d_model/n_head)
    wo = (d_model, d_model)
    """
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. q, k, v 계산
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 차원 맞추기 
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. self-attention 실행
        attention_ouput, attention_score = self.attention(q, k, v, mask=mask)

        # 4. attention output concat 및 가중치 학습
        attention_ouput = self.concat(attention_ouput)
        attention_ouput = self.w_concat(attention_ouput)

        return attention_ouput


    def split(self, tensor):
        """
        q, k, v를 head 별로 분할
        입력되는 q, k, v의 차원: [batch_size, n_seq, d_model]
        출력되는 q, k, v의 차원: [batch_size, self.head, n_seq, d_v]
        """
        batch_size, n_seq, d_model = tensor.size()

        d_v = d_model // self.n_head    # q, k, v의 마지막 차원
        tensor = tensor.view(batch_size, n_seq, self.n_head, d_v).transpose(1, 2)   # 한번에 계산했던거를 나누기
        
        return tensor
    
    def concat(self, tensor):
        """
        head별로 분할되어 계산된 dv들을 concat후 tensor 차원 다시 [batch_size, n_seq, d_model]로 맞춤
        * contiguous: tensor를 메모리에 연속적인 형태로 저장
        """
        batch_size, n_head, n_seq, d_v = tensor.size()
        d_model = d_v*n_head

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, n_seq, d_model)
        return tensor
