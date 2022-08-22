from torch import nn
import torch
from torch.nn import functional as F
import sys

class Mapping(nn.Module):
    def __init__(self,hidden_dim,common_size,data,helper,config) -> None:
        super(Mapping, self).__init__()
        dg_dg = data.dg_dg
        dg_ds = data.dg_ds
        dg_se = data.dg_se
        pt_ds = data.pt_ds
        pt_pt = data.pt_pt
        self.dg_dg = helper.to_floattensor(dg_dg, config.use_gpu)
        self.dg_ds = helper.to_floattensor(dg_ds, config.use_gpu)
        self.dg_se = helper.to_floattensor(dg_se, config.use_gpu)
        self.pt_ds = helper.to_floattensor(pt_ds, config.use_gpu)
        self.pt_pt = helper.to_floattensor(pt_pt, config.use_gpu)
        self.ds_emb = nn.Parameter(torch.FloatTensor(
            config.ds_nums, config.hidden_dim), requires_grad=True)  # 疾病数量*common_size
        self.se_emb = nn.Parameter(torch.FloatTensor(
            config.se_nums, config.hidden_dim), requires_grad=True)
        #正态分布填充数据
        self.ds_emb.data.normal_(0, 0.01)
        self.se_emb.data.normal_(0, 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, common_size)
        )
        for model in self.mlp:
            if isinstance(model, nn.Linear):  # isinstance（a,type）如果a是type类型，返回true
                nn.init.xavier_normal_(model.weight, gain=1.414)
    def forward(self,helper,en_d_embedding,en_p_embedding,all_tag,all_dg_index,all_pt_index):
        # #映射在同一个公共空间
        dg_common = self.mlp(en_d_embedding)
        pt_common = self.mlp(en_p_embedding)
        ds_common = self.mlp(self.ds_emb)
        se_common = self.mlp(self.se_emb)
        distance_loss = helper.comput_distance_loss(dg_common,pt_common,se_common,ds_common,self.dg_dg,self.dg_se,self.dg_ds,self.pt_pt,self.pt_ds,all_tag,all_dg_index,all_pt_index)
        return distance_loss
        