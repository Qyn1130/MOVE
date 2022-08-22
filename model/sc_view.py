import sys
from random import sample
import numpy as np
from torch import nn, tensor
import torch
from torch.nn import functional as F
from config import Config

class inter_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(inter_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
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
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


class intra_att(nn.Module):     
    def __init__(self, hidden_dim, attn_drop):
        super(intra_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):  
        nei_emb = F.embedding(nei, h)    
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)   
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)    
        attn_curr = self.attn_drop(self.att)              
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)           
        nei_emb = (att*nei_emb).sum(dim=1)   
        return nei_emb

class SC_encoder(nn.Module):
    def __init__(self,d_nei_num,p_nei_num,hidden_dim,attn_drop) -> None:
        super(SC_encoder, self).__init__()
        config = Config()
        self.intra_d = nn.ModuleList([intra_att(hidden_dim,attn_drop) for _ in range(d_nei_num)])
        self.intra_p = nn.ModuleList([intra_att(hidden_dim,attn_drop) for _ in range(p_nei_num)])
        self.inter_d = inter_att(hidden_dim,attn_drop)
        self.inter_p = inter_att(hidden_dim,attn_drop)
        self.d_nei_num = d_nei_num
        self.p_nei_num = p_nei_num
        self.sample_rate_d = config.sample_rate_d
        self.sample_rate_p = config.sample_rate_p
        self.gpu = config.gpu
    
    def forward(self,h,Nei_d_index,Nei_p_index):
        config = Config()
        embeds = []
        for i in range(self.d_nei_num):
            sele_nei = []
            sample_num = self.sample_rate_d[i]
            for per_node_nei in Nei_d_index[i]:
                select_one = []
                if len(per_node_nei) == 0:
                    for _ in range(sample_num):
                        select_one.append(len(h[i]))
                    select_one = torch.tensor(select_one)[np.newaxis]
                elif len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei,sample_num,replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda('cuda:'+str(self.gpu))
            temp_1 = torch.zeros([1,config.hidden_dim]).cuda('cuda:'+str(self.gpu))
            temp_h = torch.cat([h[i],temp_1], axis=0).cuda('cuda:'+str(self.gpu))
            one_type_emb = F.elu(self.intra_d[i](sele_nei,temp_h,h[0]))
            embeds.append(one_type_emb)
        z_d_sc = self.inter_d(embeds)
        
        embeds = []
        order = [0,1,3]
        for i in range(self.p_nei_num):
            sele_nei = []
            sample_num = self.sample_rate_p[i]
            for per_node_nei in Nei_p_index[i]:
                select_one = []
                if len(per_node_nei) == 0:
                    for _ in range(sample_num):
                        select_one.append(len(h[order[i]]))
                    select_one = torch.tensor(select_one)[np.newaxis]
                elif len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei,sample_num,replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num,replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda('cuda:'+str(config.gpu))
            temp_1 = torch.zeros([1,config.hidden_dim]).cuda('cuda:'+str(config.gpu))
            temp_h = torch.cat([h[order[i]],temp_1], axis=0).cuda('cuda:'+str(config.gpu))
            one_type_emb = F.elu(self.intra_p[i](sele_nei,temp_h,h[1]))
            embeds.append(one_type_emb)
        z_p_sc = self.inter_p(embeds)
        return z_d_sc,z_p_sc
