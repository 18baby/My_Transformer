# transformer embedding 생성

from torch import nn 

from embedding.positional_encoding import PositionalEncoding
from embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    transformerembedding = pos_enb + token_enb   -> 최종 transformer input 생성
    """
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__

        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb
        pos_emb = self.pos_emb
        return self.drop_out(tok_emb + pos_emb)    # dropout 적용한 채로 (tok_emb + pos_emb) 결합

