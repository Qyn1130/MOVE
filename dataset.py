import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
import scipy.sparse as sp
import torch as th

from config import Config

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64


class Dataset():
    def __init__(self):
        config = Config()
        self.repeat_nums = config.repeat_nums
        self.fold_nums = config.fold_nums
        self.neg_samp_ratio = config.neg_samp_ratio

        self.dg_smi_path = config.dg_smiles_path
        self.pt_fas_path = config.pt_fasta_path
        self.smi_max_len = config.smiles_max_len
        self.fas_max_len = config.fasta_max_len

        self.dg_pt_path = config.dg_pt_path

        self.dg_dg_path = config.dg_dg_path
        self.dg_ds_path = config.dg_ds_path
        self.dg_se_path = config.dg_se_path
        self.pt_ds_path = config.pt_ds_path
        self.pt_pt_path = config.pt_pt_path

        self.charseqset = CHARPROTSET
        self.charseqset_size = config.charseqset_size

        self.charsmiset = CHARISOSMISET
        self.charsmiset_size = config.charsmiset_size

        self.ds_nums = config.ds_nums
        self.se_nums = config.se_nums
        self.dg_nums = config.dg_nums
        self.pt_nums = config.pt_nums

        self.read_data()  
        self.pre_process()  

    def read_data(self):
        # sequence data
        self.drug_smi = pd.read_csv(
            self.dg_smi_path, header=None, index_col=None).values  
        self.protein_fas = pd.read_csv(
            self.pt_fas_path, header=None, index_col=None).values  
        self.dg_pt = pd.read_csv(self.dg_pt_path, header=0, index_col=0).values

        # Load heterogeneous information
        self.dg_dg = pd.read_csv(self.dg_dg_path, header=0, index_col=0).values
        self.dg_ds = pd.read_csv(self.dg_ds_path, header=0, index_col=0).values
        self.dg_se = pd.read_csv(self.dg_se_path, header=0, index_col=0).values
        self.pt_ds = pd.read_csv(self.pt_ds_path, header=0, index_col=0).values
        self.pt_pt = pd.read_csv(self.pt_pt_path, header=0, index_col=0).values

    def pre_process(self): 
        """
        :return:all_data_set:list    repeat_nums*fold_nums*3
        """
        self.all_data_set = []
        whole_positive_index = []  
        whole_negetive_index = []  

        for i in range(self.dg_pt.shape[0]):
            for j in range(self.dg_pt.shape[1]):
                if int(self.dg_pt[i, j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(self.dg_pt[i, j]) == 0:
                    whole_negetive_index.append([i, j])
 
        for x in range(self.repeat_nums):
            negative_sample_index = np.random.choice(np.arange(len(whole_negetive_index)),
                                                     size=len(whole_positive_index), replace=False)
            data_set = np.zeros(
                (len(whole_positive_index) + len(negative_sample_index), 3), dtype=int)

            count = 0
            for item in whole_positive_index:
                data_set[count][0] = item[0]
                data_set[count][1] = item[1]
                data_set[count][2] = 1
                count = count + 1
            for i in negative_sample_index:
                data_set[count][0] = whole_negetive_index[i][0]
                data_set[count][1] = whole_negetive_index[i][1]
                data_set[count][2] = 0
                count = count + 1

            all_fold_dataset = []
            rs = np.random.randint(0, 1000, 1)[0] 
            kf = StratifiedKFold(n_splits=self.fold_nums,
                                 shuffle=True, random_state=rs)
            for train_index, test_index in kf.split(data_set[:, 0:2], data_set[:, 2]):
                train_data, test_data = data_set[train_index], data_set[test_index]
                train_data, valid_data = train_test_split(
                    train_data, test_size=0.05, random_state=rs)
                one_fold_dataset = []
                one_fold_dataset.append(train_data)
                one_fold_dataset.append(valid_data)
                one_fold_dataset.append(test_data)
                all_fold_dataset.append(one_fold_dataset)
            self.all_data_set.append(all_fold_dataset)

        self.smi_input = np.zeros(
            (len(self.drug_smi), self.smi_max_len), dtype=int)
        self.fas_input = np.zeros(
            (len(self.protein_fas), self.fas_max_len), dtype=int)
        for i in range(len(self.drug_smi)):
            smi_len = len(self.drug_smi[i, 1]) if len(
                self.drug_smi[i, 1]) < self.smi_max_len else self.smi_max_len
            for j in range(smi_len):
                ch = self.drug_smi[i, 1][j]
                self.smi_input[i, j] = self.charsmiset[ch]

        for i in range(len(self.protein_fas)):
            fas_len = len(self.protein_fas[i, 1]) if len(
                self.protein_fas[i, 1]) < self.fas_max_len else self.fas_max_len
            for j in range(fas_len):
                ch = self.protein_fas[i, 1][j]
                self.fas_input[i, j] = self.charseqset[ch]

    def get_train_batch(self, repeat_nums, flod_nums, batch_size):
        train_drugs = []
        train_proteins = []
        train_affinity = []
        drug_index = []
        protein_index = []
        train_data = self.all_data_set[repeat_nums][flod_nums][0]

        for index, (i, j, tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)
            drug_index.append(i)
            protein_index.append(j)

        train_drugs = np.array(train_drugs) 
        train_proteins = np.array(train_proteins)  
        train_affinity = np.array(train_affinity)
        drug_index = np.array(drug_index)
        protein_index = np.array(protein_index)

        data_index = np.arange(len(train_drugs))
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]
        drug_index = drug_index[data_index]
        protein_index = protein_index[data_index]

        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex, :]
            tra_pt_batch = train_proteins[sindex:eindex, :]
            tra_tag_batch = train_affinity[sindex:eindex]
            dg_index_batch = drug_index[sindex:eindex]
            pt_index_batch = protein_index[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:, :]
            tra_pt_batch = train_proteins[sindex:, :]
            tra_tag_batch = train_affinity[sindex:]
            dg_index_batch = drug_index[sindex:]
            pt_index_batch = protein_index[sindex:]
            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

    def get_valid_batch(self, repeat_nums, flod_nums, batch_size):
        train_drugs = []
        train_proteins = []
        train_affinity = []
        drug_index = []
        protein_index = []
        # print(self.all_data_set)
        train_data = self.all_data_set[repeat_nums][flod_nums][1]

        for index, (i, j, tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)
            drug_index.append(i)
            protein_index.append(j)

        train_drugs = np.array(train_drugs)  
        train_proteins = np.array(train_proteins) 
        train_affinity = np.array(train_affinity)
        drug_index = np.array(drug_index)
        protein_index = np.array(protein_index)


        data_index = np.arange(len(train_drugs))
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]
        drug_index = drug_index[data_index]
        protein_index = protein_index[data_index]


        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex, :]
            tra_pt_batch = train_proteins[sindex:eindex, :]
            tra_tag_batch = train_affinity[sindex:eindex]
            dg_index_batch = drug_index[sindex:eindex]
            pt_index_batch = protein_index[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp

            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:, :]
            tra_pt_batch = train_proteins[sindex:, :]
            tra_tag_batch = train_affinity[sindex:]
            dg_index_batch = drug_index[sindex:]
            pt_index_batch = protein_index[sindex:]
            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

    def get_test_batch(self, repeat_nums, flod_nums, batch_size):

        train_drugs = []
        train_proteins = []
        train_affinity = []
        drug_index = []
        protein_index = []
        train_data = self.all_data_set[repeat_nums][flod_nums][2]

        for index, (i, j, tag) in enumerate(train_data):
            train_drugs.append(self.smi_input[i])
            train_proteins.append(self.fas_input[j])
            train_affinity.append(tag)
            drug_index.append(i)
            protein_index.append(j)

        train_drugs = np.array(train_drugs)
        train_proteins = np.array(train_proteins)
        train_affinity = np.array(train_affinity)
        drug_index = np.array(drug_index)
        protein_index = np.array(protein_index)

        data_index = np.arange(len(train_drugs))
        np.random.shuffle(data_index)
        train_drugs = train_drugs[data_index]
        train_proteins = train_proteins[data_index]
        train_affinity = train_affinity[data_index]
        drug_index = drug_index[data_index]
        protein_index = protein_index[data_index]

        sindex = 0
        eindex = batch_size
        while eindex < len(train_drugs):
            tra_dg_batch = train_drugs[sindex:eindex, :]
            tra_pt_batch = train_proteins[sindex:eindex, :]
            tra_tag_batch = train_affinity[sindex:eindex]
            dg_index_batch = drug_index[sindex:eindex]
            pt_index_batch = protein_index[sindex:eindex]

            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

        if eindex >= len(train_drugs):
            tra_dg_batch = train_drugs[sindex:, :]
            tra_pt_batch = train_proteins[sindex:, :]
            tra_tag_batch = train_affinity[sindex:]
            dg_index_batch = drug_index[sindex:]
            pt_index_batch = protein_index[sindex:]
            yield tra_dg_batch, tra_pt_batch, tra_tag_batch, dg_index_batch, pt_index_batch

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""

        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = th.from_numpy( 
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  
        values = th.from_numpy(sparse_mx.data)
        shape = th.Size(sparse_mx.shape)
        return th.sparse.FloatTensor(indices, values, shape)

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))  
        r_inv = np.power(rowsum, -1).flatten()  
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)  
        features = r_mat_inv.dot(features)  
        return features.todense()  

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))  
        d_inv_sqrt = np.power(rowsum+1e-8, -0.5).flatten()  
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def get_data_for_HIN(self):
        feat_dg = sp.eye(self.dg_nums)
        feat_pt = sp.eye(self.pt_nums)
        feat_se = sp.eye(self.se_nums)
        feat_ds = sp.eye(self.ds_nums)

        feat_dg = th.FloatTensor(self.preprocess_features(feat_dg))
        feat_pt = th.FloatTensor(self.preprocess_features(feat_pt))
        feat_se = th.FloatTensor(self.preprocess_features(feat_se))
        feat_ds = th.FloatTensor(self.preprocess_features(feat_ds))

        dg_ds_dg = sp.load_npz("data/drug_disease_drug.npz")
        dg_pt_dg = sp.load_npz("data/drug_protein_drug.npz")
        dg_se_dg = sp.load_npz("data/drug_se_drug.npz")
        dg_ds_dg = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adj(dg_ds_dg))
        dg_pt_dg = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adj(dg_pt_dg))
        dg_se_dg = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adj(dg_se_dg))

        pt_ds_pt = sp.load_npz("data/protein_disease_protein.npz")
        pt_dg_pt = sp.load_npz("data/protein_drug_protein.npz")
        pt_ds_pt = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adj(pt_ds_pt))
        pt_dg_pt = self.sparse_mx_to_torch_sparse_tensor(
            self.normalize_adj(pt_dg_pt))

        dg_nei_dg = np.load("data/drug_drug_neibor.npy", allow_pickle=True)
        dg_nei_pt = np.load("data/drug_protein_neibor.npy", allow_pickle=True)
        dg_nei_se = np.load("data/drug_se_neibor.npy", allow_pickle=True)
        dg_nei_ds = np.load("data/drug_disease_neibor.npy", allow_pickle=True)
        dg_nei_dg = [th.LongTensor(i) for i in dg_nei_dg]
        dg_nei_pt = [th.LongTensor(i) for i in dg_nei_pt]
        dg_nei_se = [th.LongTensor(i) for i in dg_nei_se]
        dg_nei_ds = [th.LongTensor(i) for i in dg_nei_ds]


        pt_nei_dg = np.load("data/protein_drug_neibor.npy", allow_pickle=True)
        pt_nei_pt = np.load(
            "data/protein_protein_neibor.npy", allow_pickle=True)
        pt_nei_ds = np.load(
            "data/protein_disease_neibor.npy", allow_pickle=True)
        pt_nei_dg = [th.LongTensor(i) for i in pt_nei_dg]
        pt_nei_pt = [th.LongTensor(i) for i in pt_nei_pt]
        pt_nei_ds = [th.LongTensor(i) for i in pt_nei_ds]

        return [feat_dg, feat_pt, feat_se, feat_ds], [dg_ds_dg, dg_pt_dg, dg_se_dg], [pt_ds_pt, pt_dg_pt], [dg_nei_dg, dg_nei_pt, dg_nei_se, dg_nei_ds], [pt_nei_dg, pt_nei_pt, pt_nei_ds]
