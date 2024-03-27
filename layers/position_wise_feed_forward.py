# 두번째 fully connected 층

from torch import nn

class PositionWiseFeedFoward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionWiseFeedFoward, self).__init__()
        self.l1 = nn.Linear(d_model, hidden)
        self.l2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.l2(x)
        return x
