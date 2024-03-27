# positional encodeing 진행

import torch
from torch import nn 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        """
        d_model: 각 단어 임베딩 크기
        max_len: 입력 sequence의 최대 길이
        """

        self.encoding = torch.zeros(max_len, d_model, device=device)  # 인코딩 결과 저장
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)    # 순서 규정
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()   # 짝수 위치 규정

        # 인코딩 진행
        self.encoding[:, 0::2] = torch.sin(pos/(10000)**(_2i/d_model))
        self.encoding[:, 1::2] = torch.cos(pos/(10000)**(_2i/d_model))


    def forward(self, x):
        """
        max_len = 512
        d_model = 512
        batch_size = 128
        seq_len = 30
        """
        batch_size, seq_len = x.size()

        return self.encoding[:seq_len, :]



