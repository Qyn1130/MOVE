from torch import nn
import torch
from torch.nn import functional as F

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)   #初始化一个[1,hidden_dim]的矩阵
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp

class MP_encoder(nn.Module):
    def __init__(self, P_d,P_p,hidden_dim,attn_drop) -> None:
        super(MP_encoder, self).__init__()
        self.P_d = P_d
        self.P_p = P_p
        self.dg_mp = nn.ModuleList([GCN(hidden_dim,hidden_dim) for _ in range(len(P_d))])
        self.pt_mp = nn.ModuleList([GCN(hidden_dim,hidden_dim) for _ in range(len(P_d))])
        self.dg_att = Attention(hidden_dim,attn_drop)
        self.pt_att = Attention(hidden_dim,attn_drop)
    
    
    def forward(self,h_d,h_p, mps_d, mps_p):
        embeds_d = []
        embeds_p = []
        for i in range(len(self.P_d)):
            embeds_d.append(self.dg_mp[i](h_d,mps_d[i]))
        z_d_mp = self.dg_att(embeds_d)
        for i in range(len(self.P_p)):
            embeds_p.append(self.pt_mp[i](h_p,mps_p[i]))
        z_p_mp = self.pt_att(embeds_p)
        return z_d_mp,z_p_mp