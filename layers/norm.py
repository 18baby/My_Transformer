import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # 정규화 가중치
        self.beta = nn.Parameter(torch.zeros(d_model))   # 정규화 편향
        self.eps = eps

    def forward(self, x):
        """
        x : [batch_size, n_seq, d_model]
        '-1'은 마지막 차원을 의미(단어 별로 평균, 분산 계산)
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        # 정규화 진행
        out = (x-mean)/torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
