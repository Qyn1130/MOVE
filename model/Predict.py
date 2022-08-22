from torch import nn, sigmoid
import torch
from torch.nn import functional as F

class Predict(nn.Module,):
    def __init__(self, hidden_dim, common_size) -> None:
        super(Predict, self).__init__()
        self.mlp = MLP(hidden_dim*2, 1)

    def forward(self, smi_common, fas_common):
        commom = self.mlp(torch.cat((smi_common, fas_common), 1))
        return commom


class MLP(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )
        for model in self.linear:
            if isinstance(model, nn.Linear):  # isinstance（a,type）如果a是type类型，返回true
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def forward(self, x):
        bs = len(x)
        out = self.linear(x)
        return out.reshape(bs)
