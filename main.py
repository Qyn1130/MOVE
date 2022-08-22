from audioop import avg
import time
import datetime
import os

import numpy as np
from sklearn import datasets
import torch.optim as optim
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from config import Config
from model.DTI import DTI
from model.mapping import Mapping
from model.Predict import Predict
from utils.util import Helper
from dataset import Dataset
final_results = []

all_result = []


def train_DTI(model, config, helper, data, repeat_nums, fold_nums, feats, P_d, P_p, epoch):
    fold_begin_time = time.time()
    optimizer = optim.Adam(
        model.parameters(), config.contrast_learn_rate, weight_decay=1e-8)
    model.train()

    print("DTI Model begin training------------------------------",
          datetime.datetime.now())
    all_loss = 0
    for e in range(config.contrast_epochs):
        batch_num = 0
        for i, (dg, pt, tag, dg_index, pt_index) in enumerate(data.get_train_batch(repeat_nums, fold_nums, config.batch_size)):  # (dg_index,pt_index,tag)
            optimizer.zero_grad()
            pred, loss = model(config, feats, P_d, P_p, dg, pt,
                               tag, dg_index, pt_index, helper)
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            batch_num = batch_num + 1
        print("the loss of DTI model epoch[%d/%d]:is %4.f,time:%d s" % (e+1, config.contrast_epochs, all_loss, time.time()-fold_begin_time))
    return all_loss


def valid_model(onfig, helper, model, data, repeat_nums, fold_nums, feats, P_d, P_p):
    model.eval()
    print("evaluate the model")

    begin_time = time.time()
    loss = 0
    avg_acc = []
    avg_auc = []
    avg_aupr = []
    avg_pre = []
    avg_f1 = []
    avg_recall = []

    with torch.no_grad():
        for i, (dg, pt, tag, dg_index, pt_index) in enumerate(data.get_valid_batch(repeat_nums, fold_nums, config.batch_size)):
            pred = model.get_tag(config,  dg, pt, dg_index, pt_index,feats, P_d, P_p)
            try:
                pred = pred.cpu()
                auc = roc_auc_score(tag, pred)
                aupr = average_precision_score(tag, pred)

                pred = [1 if x > 0.5 else 0 for x in pred]
                acc = accuracy_score(tag, pred)
                pre = precision_score(tag, pred)
                recall = recall_score(tag, pred)

                f1 = f1_score(tag, pred)

                avg_acc.append(acc)
                avg_pre.append(pre)
                avg_recall.append(recall)
                avg_f1.append(f1)
                avg_auc.append(auc)
                avg_aupr.append(aupr)
            except ValueError:
                pass
        print("valid:  avg_auc:", np.mean(avg_auc), "avg_aupr:", np.mean(avg_aupr), "avg_acc",
          np.mean(avg_acc), "precision", np.mean(avg_pre), "recall", np.mean(avg_recall), "f1_score", np.mean(avg_f1))
    return np.mean(avg_auc)

def evaluation_model(config, helper, model, data, repeat_nums, fold_nums,feats, P_d, P_p):
    model.load_state_dict(torch.load('./' + config.result_path +
                          '/DTI_model_parm/repeat_%d_corss_%d.parm' % (repeat_nums, fold_nums)))
    model.eval()
    print("test the model")

    begin_time = time.time()
    loss = 0
    avg_acc = []
    avg_auc = []
    avg_aupr = []
    avg_pre = []
    avg_f1 = []
    avg_recall = []

    with torch.no_grad():
        for i, (dg, pt, tag, dg_index, pt_index) in enumerate(data.get_test_batch(repeat_nums, fold_nums, config.batch_size)):
            # print(dg_index)
            if(len(dg_index) != 1):
                pred = model.get_tag(config,  dg, pt, dg_index, pt_index, feats, P_d, P_p)
                try:
                    pred = pred.cpu()
                    auc = roc_auc_score(tag, pred)
                    aupr = average_precision_score(tag, pred)

                    pred = [1 if x > 0.5 else 0 for x in pred]
                    acc = accuracy_score(tag, pred)
                    pre = precision_score(tag, pred)
                    recall = recall_score(tag, pred)

                    f1 = f1_score(tag, pred)

                    avg_acc.append(acc)
                    avg_pre.append(pre)
                    avg_recall.append(recall)
                    avg_f1.append(f1)
                    avg_auc.append(auc)
                    avg_aupr.append(aupr)
                except ValueError:
                    pass

    print("the total_loss of test model:is %4.f, time:%d s" %
          (loss, time.time() - begin_time))
    print("avg_auc:", np.mean(avg_auc), "avg_aupr:", np.mean(avg_aupr), "avg_acc",
          np.mean(avg_acc), "precision", np.mean(avg_pre), "recall", np.mean(avg_recall), "f1_score", np.mean(avg_f1))
    result = []
    result.append(np.mean(avg_auc))
    result.append(np.mean(avg_aupr))
    result.append(np.mean(avg_acc))
    result.append(np.mean(avg_pre))
    result.append(np.mean(avg_recall))
    result.append(np.mean(avg_f1))
    final_results.append(result)



if __name__ == '__main__':

    model_begin_time = time.time()
    config = Config()
    helper = Helper()
    data = Dataset()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    device_ids = range(torch.cuda.device_count())
    if not os.path.exists('./' + config.result_path):
        os.mkdir('./' + config.result_path)
    if not os.path.exists('./' + config.result_path + '/DTI_model_parm'):
        os.mkdir('./' + config.result_path + '/DTI_model_parm')
    all_lam = [0.25]
    all_lc = [0.16]
    all_tau = [0.5]
    
    
    feats, P_d, P_p, Nei_d, Nei_p = data.get_data_for_HIN()
    if torch.cuda.is_available():
        feats = [feat.cuda('cuda:'+str(config.gpu)) for feat in feats]
        P_d = [p.cuda('cuda:'+str(config.gpu)) for p in P_d]
        P_p = [p.cuda('cuda:'+str(config.gpu)) for p in P_p]
    feats_dim_list = [i.shape[1] for i in feats]

    for i in range(config.repeat_nums):
        for lam in all_lam:
            config.set_lam(lam)
            for tau in all_tau:
                config.set_tau(tau)
                for lc in all_lc:
                    config.set_l_c(lc)
                    final_results = []
                    print("lc=",config.l_c)
                    print("tau=",config.tau)
                    print("lam=",config.lam)
                    for j in range(config.fold_nums): 
                        fold_begin_time = time.time()
                        print("crossfold", j+1, "++++++++++++++++++++++++++++++++++++++++++++")
                        cnt_wait = 0
                        best = 0
                        epoch_loss = 0.0
                        model = DTI(feats_dim_list, P_p, P_d, Nei_d, Nei_p, data, helper)
                        if torch.cuda.is_available():
                            model.cuda('cuda:'+str(config.gpu))
                        for epoch in range(config.num_epochs):
                            print("         epoch:", str(epoch), "zzzzzzzzzzzzzzzz")
                            epoch_loss = train_DTI(model, config, helper, data, i, j, feats, P_d, P_p, epoch)
                            print("fold", j, "epoch_loss = ", epoch_loss)
                            
                            auc = valid_model(config, helper, model,data, i, j, feats, P_d, P_p)
                            if auc > best:
                                best = auc
                                best_t = epoch
                                cnt_wait = 0
                                torch.save(model.state_dict(), './' + config.result_path +
                                        '/DTI_model_parm/repeat_%d_corss_%d.parm' % (i, j))
                            else:
                                cnt_wait += 1

                            if cnt_wait == 3:
                                print("Early Stopping!")
                                evaluation_model(config, helper, model,data, i, j, feats, P_d, P_p)
                                break
                        if epoch == config.num_epochs - 1:
                            evaluation_model(config, helper, model,data, i, j, feats, P_d, P_p)


                    avg_results = np.sum(final_results, axis=0)/len(final_results)
                    print("model avg_auc:", avg_results[0])
                    print("model avg_aupr:", avg_results[1])
                    print("model avg_acc", avg_results[2])
                    print("model avg_pre", avg_results[3])
                    print("model avg_recall", avg_results[4])
                    print("model avg_f1", avg_results[5])
                    avg_results = np.append(avg_results,lc)
                    avg_results = np.append(avg_results,tau)
                    avg_results = np.append(avg_results,lam)

                    all_result.append(avg_results)
    
    result_file1 = pd.DataFrame(all_result)
    result_file1.to_csv(config.result_path + '/result.csv', mode='a',
                    index=False, header=False, float_format='%.3f', encoding='utf-8')