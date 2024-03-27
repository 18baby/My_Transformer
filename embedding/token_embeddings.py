from torch import nn


class TokenEmbedding(nn.Embedding):

    def __init__(self, vocab_size, d_model):
        """
        vocab_size: 단어사전 크기
        d_model: 입력 단어 차원
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)