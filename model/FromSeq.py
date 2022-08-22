import sys
from torch import nn
import torch
from torch.nn import functional as F
from config import Config


#处理SMILES和氨基酸序列
class FromSeq(nn.Module):
    def __init__(self) -> None:
        super(FromSeq, self).__init__()
        
        config = Config()
        self.smi_emb = nn.Embedding(config.charsmiset_size+1, config.embedding_size)
        self.smi_conv1 = nn.Conv1d(1,config.num_filters,(config.smi_window_lengths,config.embedding_size),stride=1,padding=0)
        self.smi_conv2 = nn.Conv1d(config.num_filters,config.num_filters*2,(config.smi_window_lengths,1),stride=1,padding=0)
        self.smi_conv3 = nn.Conv1d(config.num_filters*2,config.hidden_dim,(config.smi_window_lengths,1),stride=1,padding=0)
        self.smi_maxpool = nn.MaxPool2d(kernel_size=(1,141))
        
        
        #处理氨基酸序列
        self.fas_emb = nn.Embedding(config.charseqset_size+1, config.embedding_size)
        self.fas_conv1 = nn.Conv1d(1,config.num_filters,(config.fas_window_lengths,config.embedding_size),stride=1,padding=0)
        self.fas_conv2 = nn.Conv1d(config.num_filters,config.num_filters*2,(config.fas_window_lengths,1),stride=1,padding=0)
        self.fas_conv3 = nn.Conv1d(config.num_filters*2,config.hidden_dim,(config.fas_window_lengths,1),stride=1,padding=0)
        self.fas_maxpool = nn.MaxPool2d(kernel_size=(1,1479))
        
    
    
    def forward(self,smiles,fasta):
        smiles_vector = self.smi_emb(smiles)   #in[batch_size,max_smile_length] out[batch_size,max_smile_length,embedding_size]=[batch, 150, 128]
        smiles_vector = torch.unsqueeze(smiles_vector,1) #out[batch, 1, 150, 128]
        smiles_vector = self.smi_conv1(smiles_vector)     #out[batch, num_filters, 147, 1]   
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.smi_conv2(smiles_vector)   #[batch,num_filters*2, 144, 1]
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.smi_conv3(smiles_vector)  # #[batch, num_filters*2, 141, 1]
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = smiles_vector.squeeze()         #[batch, num_filters*2, 141]
        smiles_vector = self.smi_maxpool(smiles_vector)  ##[batch,num_filters*2,1]
        smile_seq = smiles_vector.squeeze()             # out[batch,num_filters*2]
        
        fasta_vector = self.fas_emb(fasta)          #in[batch_size,max_fas_length]  out[batch_size,max_fasta_length,embedding_size] = [batch,1500,128]
        fasta_vector = torch.unsqueeze(fasta_vector,1)    #out[batch,1,1500,128]
        fasta_vector = self.fas_conv1(fasta_vector)  #out[batch,num_filters,1493,1]
        fasta_vector = torch.relu(fasta_vector)     
        fasta_vector = self.fas_conv2(fasta_vector)  #out[batch,num_filters*2,1486,1] 
        fasta_vector = torch.relu(fasta_vector)      
        fasta_vector = self.fas_conv3(fasta_vector)  #out[batch,num_filters*2,1479,1]
        fasta_vector = fasta_vector.squeeze()         #out[batch,num_filters*2,1479]
        fasta_vector = self.fas_maxpool(fasta_vector)   #[batch,num_filters*2,1]
        fasta_seq = fasta_vector.squeeze()       # out[batch,num_filters*2]
        return smile_seq,fasta_seq
    
        
        
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out
        
        