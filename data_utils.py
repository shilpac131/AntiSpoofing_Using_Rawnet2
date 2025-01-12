import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

def spklevel_utterence(dev_loader, model, device):
    i = 0
    utterence_spk = []
    for batch_x, batch_y in dev_loader:
        batch_x = batch_x.to(device)
        torch.cuda.empty_cache()
        output = model(batch_x)
        output = output.squeeze()
        utterence_spk.append(output)
        del output
        torch.cuda.empty_cache()
        print("calculating embedding for batch ",(i+1))
        i = i+1
    return utterence_spk

def genSpoof_list_spkwise( dir_meta , file_limit,spk_id,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list = []
    
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
            if file_limit is not None and len(file_list) >= file_limit:
                break  # Stop processing files if the limit is reached
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
            if file_limit is not None and len(file_list) >= file_limit:
                break  # Stop processing files if the limit is reached
        return file_list

    else:
        for line in l_meta:
            spk_info, key, _, _, label = line.strip().split(' ')
            if spk_info == spk_id and label == 'spoof':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
                if file_limit is not None and len(file_list) >= file_limit:
                    break  # Stop processing files if the limit is reached
        return d_meta, file_list

def genSpoof_list(dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:   
        for line in l_meta:
             _, key,_,_,label = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def genSpoof_list_ASVspoof21( dir_meta,is_train=False,is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _, key,_,_,_,label,_,_,_,_,_,_,_ = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
             _, key,_,_,_,label,_,_,_,_,_,_,_  = line.strip().split(' ')
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y
            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,key      



class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs    : list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            self.labels = labels
            

    def __len__(self):
            return len(self.list_IDs)


    def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+key+'.flac', sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels[key]
            return x_inp, y      
           
            
            

                
                
                



