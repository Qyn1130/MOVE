import this
from torch import batch_norm, mul, nn
import torch
from torch.nn import functional as F
import numpy as np
import sys
from config import Config


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam, batch_size) -> None:
        super(Contrast, self).__init__()
        self.config = Config()
        self.tau = tau
        self.lam = lam
        self.batch_size = batch_size
        

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())  
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, d_seq_emb, p_seq_emb, d_HIN_emb, p_HIN_emb, neg_num):
        this_batch_size = d_seq_emb.size()[0]
        if this_batch_size == 1:
            return 0
        else:
            d_neg_matrix = torch.full((this_batch_size, this_batch_size),1.0).cuda('cuda:'+str(self.config.gpu))
            p_neg_matrix = torch.full((this_batch_size, this_batch_size),1.0).cuda('cuda:'+str(self.config.gpu))            
            
            d_matrix_seq2HIN = self.sim(d_seq_emb, d_HIN_emb)
            d_matrix_HIN2seq = d_matrix_seq2HIN.t()
            p_matrix_seq2HIN = self.sim(p_seq_emb, p_HIN_emb)
            p_matrix_HIN2seq = p_matrix_seq2HIN.t()
            pos = torch.eye(this_batch_size).cuda('cuda:'+str(self.config.gpu))
            
            d_matrix_seq2HIN = (d_matrix_seq2HIN.mul(pos).sum(dim=1)) / (torch.sum(d_matrix_seq2HIN.mul(d_neg_matrix), dim=1).view(-1, 1)+1e-8)
            # print("d_matrix_seq2HIN",d_matrix_seq2HIN)
            d_loss1 = -torch.log(d_matrix_seq2HIN).mean()

            d_matrix_HIN2seq = (d_matrix_HIN2seq.mul(pos).sum(dim=1)) / (torch.sum(d_matrix_HIN2seq.mul(d_neg_matrix), dim=1).view(-1, 1)+1e-8)
            # print("d_matrix_HIN2seq",d_matrix_HIN2seq)
            d_loss2 = -torch.log(d_matrix_HIN2seq).mean()

            p_matrix_seq2HIN = (p_matrix_seq2HIN.mul(pos).sum(dim=1)) / (torch.sum(p_matrix_seq2HIN.mul(p_neg_matrix), dim=1).view(-1, 1)+1e-8)
            # print("p_matrix_seq2HIN",p_matrix_seq2HIN)
            p_loss1 = -torch.log(p_matrix_seq2HIN).mean()

            p_matrix_HIN2seq = (p_matrix_HIN2seq.mul(pos).sum(dim=1)) / (torch.sum(p_matrix_HIN2seq.mul(p_neg_matrix), dim=1).view(-1, 1)+1e-8)
            # print("p_matrix_HIN2seq",p_matrix_HIN2seq)
            p_loss2 = -torch.log(p_matrix_HIN2seq).mean()
            contrast_loss = self.lam*d_loss1 + (0.5 - self.lam) * d_loss2 + self.lam * p_loss1 + (0.5 - self.lam) * p_loss2
        return contrast_loss 
