from config import Config
import sys
from sklearn import feature_selection
from torch import nn
import torch
from torch.nn import functional as F

from model.mp_view import MP_encoder
from model.sc_view import SC_encoder


class FromHIN(nn.Module):
    def __init__(self, feats_dim_list, hidden_dim, feat_drop, P_d, P_p, att_drop) -> None:
        super(FromHIN, self).__init__()
        config = Config()
        self.hidden_dim = hidden_dim
        self.d_nei_num = config.Nei_d_num
        self.p_nei_num = config.Nei_p_num
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = MP_encoder(P_d, P_p, hidden_dim, config.att_drop)
        self.sc = SC_encoder(
            self.d_nei_num, self.p_nei_num, hidden_dim, att_drop)

    def forward(self, feats, P_d, P_p, Nei_d_index, Nei_p_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))  
        z_mp_d, z_mp_p = self.mp(h_all[0], h_all[1], P_d, P_p)
        z_sc_d, z_sc_p = self.sc(h_all, Nei_d_index, Nei_p_index)
        z_HIN_d = z_mp_d + z_sc_d
        z_HIN_p = z_mp_p + z_sc_p
        return z_HIN_d, z_HIN_p
