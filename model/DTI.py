import sys
from torch import double, float32, nn, t
import torch
from torch.nn import functional as F
import numpy as np
import time
torch.set_printoptions(profile="full")

from model.FromSeq import FromSeq
from model.FromHIN import FromHIN
from model.mapping import Mapping
from model.Predict import Predict
from model.contrast import Contrast
from config import Config
from dataset import Dataset
import torch.optim as optim
from utils.util import Helper

# our model


class DTI(nn.Module):
    def __init__(self, feats_dim_list, P_p, P_d, Nei_d, Nei_p,data,helper) -> None:
        super(DTI, self).__init__()
        config = Config()
        self.helper = Helper()
        self.P_d = P_d
        self.P_p = P_p
        self.Nei_d = Nei_d
        self.Nei_p = Nei_p
        self.neg_num = config.neg_num
        self.seq = FromSeq()  # seq的embedding
        self.HIN = FromHIN(feats_dim_list, config.hidden_dim,
                           config.HIN_feat_drop, P_d, P_p, config.att_drop)  # HIN的embedding
        self.contrast = Contrast(
            config.hidden_dim, config.tau, config.lam, config.batch_size)
        # self.mapping = Mapping(config.hidden_dim,config.common_size,data,helper,config)
        self.predict = Predict(config.hidden_dim, config.common_size)
        
        self.proj_d = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.proj_p = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        for model in self.proj_d:
            if isinstance(model, nn.Linear):  # isinstance（a,type）如果a是type类型，返回true
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.proj_p:
            if isinstance(model, nn.Linear):  # isinstance（a,type）如果a是type类型，返回true
                nn.init.xavier_normal_(model.weight, gain=1.414)
        
        self.fusion_dg = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim))
                
        self.fusion_pt = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(True),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim))
        
        for model in self.fusion_dg:
            if isinstance(model, nn.Linear):  
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.fusion_pt:
            if isinstance(model, nn.Linear): 
                nn.init.xavier_normal_(model.weight, gain=1.414)
   

    def forward(self, config, feats, P_d, P_p, dg, pt, tag, dg_index, pt_index, helper):
        all_d_HIN_embedding, all_p_HIN_embedding = self.HIN(feats, P_d, P_p, self.Nei_d, self.Nei_p)
        d_HIN_embedding = all_d_HIN_embedding[dg_index, :]
        p_HIN_embedding = all_p_HIN_embedding[pt_index, :]
        dg = self.helper.to_longtensor(dg, config.use_gpu)
        pt = self.helper.to_longtensor(pt, config.use_gpu)
        smi_embedding, fas_embedding = self.seq(dg, pt)
        d_seq_emb = self.proj_d(smi_embedding)
        p_seq_emb = self.proj_p(fas_embedding)
        d_HIN_emb = self.proj_d(d_HIN_embedding)
        p_HIN_emb = self.proj_p(p_HIN_embedding)
        contrast_loss = self.contrast(d_seq_emb, p_seq_emb, d_HIN_emb, p_HIN_emb, self.neg_num)
        
        dg_emb = self.fusion_dg(torch.cat((d_seq_emb,d_HIN_emb),1))
        pt_emb = self.fusion_pt(torch.cat((p_seq_emb,p_HIN_emb),1))
        tag = helper.to_floattensor(tag, config.use_gpu)
        pred = self.predict(dg_emb, pt_emb)
        tagLoss = F.binary_cross_entropy(pred,tag)
        loss = config.l_c * contrast_loss + tagLoss
        return pred, loss
    
    def get_tag(self, config, dg, pt, dg_index, pt_index, feats, P_d, P_p):
        all_d_HIN_embedding, all_p_HIN_embedding = self.HIN(feats, P_d, P_p, self.Nei_d, self.Nei_p)
        d_HIN_embedding = all_d_HIN_embedding[dg_index, :]
        p_HIN_embedding = all_p_HIN_embedding[pt_index, :]
        dg = self.helper.to_longtensor(dg, config.use_gpu)
        pt = self.helper.to_longtensor(pt, config.use_gpu)
        smi_embedding, fas_embedding = self.seq(dg, pt)
        d_seq_emb = self.proj_d(smi_embedding)
        p_seq_emb = self.proj_p(fas_embedding)
        d_HIN_emb = self.proj_d(d_HIN_embedding)
        p_HIN_emb = self.proj_p(p_HIN_embedding)
        
        dg_emb = self.fusion_dg(torch.cat((d_seq_emb,d_HIN_emb),1))
        pt_emb = self.fusion_pt(torch.cat((p_seq_emb,p_HIN_emb),1))
        
        return self.predict(dg_emb, pt_emb)
        
        
    
    
