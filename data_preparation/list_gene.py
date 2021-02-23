import numpy as np
import os
import pdb
import math
import shutil
#HC:0 FMD:1 MCI:2 ND:3
#128
#62 10 29 27
'''
list_folder = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/10-fold-list/'
output_folder = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/HC_MCI_ND_lists/'
Fold = 10 

def rec_id_split(seg_num, lst, fold_idx):
    start_id = fold_idx*seg_num
    end_id = (fold_idx+1)*seg_num
    if end_id >len(lst) or fold_idx == Fold-1:
        print('final')
        end_id = len(lst)
    return lst[start_id:end_id]


ref_dict = {}
for rec_id in label_dict.keys():
    label = label_dict[rec_id]
    if label in ref_dict.keys():
        ref_dict[label].append(rec_id)
    else:


for key_id in ref_dict.keys():
    print(key_id)
    lst = ref_dict[key_id]
    seg_num = int(round(len(lst)/Fold))
    print(seg_num)
    for idx in range(Fold):
        sub_folder_path = os.path.join(output_list_folder, str(idx))
        if os.path.exists(sub_folder_path) == False:
            os.mkdir(sub_folder_path)
        if key_id == 3 and idx >=7:
            seg_num =2
        sub_lst=rec_id_split(seg_num, lst, idx)
        if key_id == 3 and idx == 9:
            sub_lst = lst[-2:]
        output_path = os.path.join(sub_folder_path, str(key_id))
        print(sub_lst)
        np.save(output_path,np.asarray(sub_lst))


def train_list (train_start_idx, train_end_idx):
    HC_lst = []
    FMD_lst = []
    MCI_lst = []
    ND_lst = []
    
    if train_start_idx>train_end_idx:
        for train_idx in range(train_start_idx, Fold):
            sub_HC_lst = np.load(os.path.join(list_folder, str(train_idx),'0.npy'))
            sub_FMD_lst = np.load(os.path.join(list_folder, str(train_idx),'1.npy'))
            sub_MCI_lst = np.load(os.path.join(list_folder, str(train_idx),'2.npy'))
            sub_ND_lst = np.load(os.path.join(list_folder, str(train_idx),'3.npy'))
            HC_lst.extend(sub_HC_lst)
            FMD_lst.extend(sub_FMD_lst)
            MCI_lst.extend(sub_MCI_lst)
            ND_lst.extend(sub_ND_lst)
        for train_idx in range(0, train_end_idx):
            sub_HC_lst = np.load(os.path.join(list_folder, str(train_idx),'0.npy'))
            sub_FMD_lst = np.load(os.path.join(list_folder, str(train_idx),'1.npy'))
            sub_MCI_lst = np.load(os.path.join(list_folder, str(train_idx),'2.npy'))
            sub_ND_lst = np.load(os.path.join(list_folder, str(train_idx),'3.npy'))
            HC_lst.extend(sub_HC_lst)
            FMD_lst.extend(sub_FMD_lst)
            MCI_lst.extend(sub_MCI_lst)
            ND_lst.extend(sub_ND_lst)
    else:
        for train_idx in range(train_start_idx, train_end_idx):
            sub_HC_lst = np.load(os.path.join(list_folder, str(train_idx),'0.npy'))
            sub_FMD_lst = np.load(os.path.join(list_folder, str(train_idx),'1.npy'))
            sub_MCI_lst = np.load(os.path.join(list_folder, str(train_idx),'2.npy'))
            sub_ND_lst = np.load(os.path.join(list_folder, str(train_idx),'3.npy'))
            HC_lst.extend(sub_HC_lst)
            FMD_lst.extend(sub_FMD_lst)
            MCI_lst.extend(sub_MCI_lst)
            ND_lst.extend(sub_ND_lst)
            
    return HC_lst, FMD_lst, MCI_lst, ND_lst
for idx in range(Fold):
    dev_idx = (idx+1)%Fold
    train_start_idx = (idx+2)%Fold
    train_end_idx = (idx+Fold)%Fold
    
    output_sub_folder = os.path.join(output_folder, str(idx))
    if os.path.exists(output_sub_folder) == False:
        os.mkdir(output_sub_folder)
    output_path =  os.path.join(output_sub_folder, 'train_list')
    lst_f = open(output_path, 'w')
    
    HC_lst, FMD_lst, MCI_lst, ND_lst = train_list (train_start_idx, train_end_idx)

    sub_lst = np.concatenate([HC_lst, MCI_lst, ND_lst], axis=0)
    print(len(sub_lst))
    for rec_id in sub_lst:
        lst_f.write(rec_id+'\n')
    lst_f.close()
    

label_dict_path = '/data/ac1yp/data/cookie_theft/bristol/HC_MCI_ND_dict.npy'
list_path = '/data/ac1yp/data/cookie_theft/bristol/HC_MCI_ND_list'
list_f = open(list_path,'w')
label_dict = np.load(label_dict_path, allow_pickle=True).item()
for rec_id in label_dict.keys():
   list_f.write(rec_id+'\n')
list_f.close()

DB_list_folder = '/data/ac1yp/data/cookie_theft/DementiaBank/list/'
output_list_folder = '/data/ac1yp/data/cookie_theft/Bristol_exclude/list/'
comb_list_path = '/data/ac1yp/data/cookie_theft/Bristol_exclude/comb_list'
comb_list = open(comb_list_path).readlines()
Fold = 10

for idx in range(Fold):
    train_path = os.path.join(DB_list_folder, str(idx), 'train_wav.scp')
    output_path = os.path.join(output_list_folder, str(idx), 'train_wav.scp')
    shutil.copyfile(train_path, output_path)
    train_f = open(output_path,'a')
    for line in comb_list:
        train_f.write(line)
    train_f.close()


list_folder = '/data/ac1yp/data/cookie_theft/DementiaBank/list/'
for idx in os.listdir(list_folder):
    sub_folder = os.path.join(list_folder, str(idx))
    for file_name in os.listdir(sub_folder):
        sub_path = os.path.join(sub_folder, file_name)
        new_name = file_name.split('_')[0]+'_list'
        if new_name == 'val_list':
            new_name = 'dev_list'
        new_path = os.path.join(sub_folder, new_name)
        os.rename(sub_path, new_path)  


list_folder = '/data/ac1yp/data/cookie_theft/lists/ADReSS_test/ADReSS_Oth./list/'  
list_path = '/data/ac1yp/data/cookie_theft/bristol/HC_ND_list'
list = open(list_path).readlines()
for sub_l in os.listdir(list_folder):
    sub_folder = os.path.join(list_folder, sub_l)
    file_path = os.path.join(sub_folder, 'train_list')
    
    f_ = open(file_path,'a')
    for rec_id in list:
        f_.write(rec_id)
    f_.close()

list_folder = '/data/ac1yp/data/cookie_theft/lists/ADReSS_test/'
list_path = '/data/ac1yp/code/Challenge_TAPAS/dataset/text/test/test_list'
for sub_folder in os.listdir(list_folder):
    sub_folder_path = os.path.join(list_folder, sub_folder, 'list')
    for idx in range(9):
        sub2_folder_path = os.path.join(sub_folder_path, str(idx))
        tar_path = os.path.join(sub2_folder_path, 'test_list')
        shutil.copyfile(list_path, tar_path)


def list_rewrite(list_path):
    lists = open(list_path).readlines()
    list_f = open(list_path,'w')
    for line in lists:
        rec_id=line.split(' ')[0]
        list_f.write(rec_id+'\n')
    list_f.close()
        

list_folder = '/data/ac1yp/data/cookie_theft/lists/ADReSS_test/' 
for sub_name in os.listdir(list_folder):
    sub_folder = os.path.join(list_folder, sub_name, 'list')
    
    for idx in os.listdir(sub_folder):
        dev_list_path = os.path.join(sub_folder, idx,'dev_list')
        train_list_path = os.path.join(sub_folder, idx, 'train_list')
        
        list_rewrite(dev_list_path)
        list_rewrite(train_list_path)
        

data_list_path = '/data/ac1yp/data/cookie_theft/Manchester/lists/binary_list' 
transcript_folder = '/data/ac1yp/data/cookie_theft/Manchester/transcript/'
data_list = open(data_list_path).readlines()


for rec_name in data_list:
    rec_name = rec_name[:-1]     
    if os.path.exists(os.path.join(transcript_folder, rec_name)) == False:
        print(rec_name)
'''


list_folder = '/data/ac1yp/data/cookie_theft/lists/DB_test/DB_IVA_ADReSS_Manchester//list/'
data_list_path = '/data/ac1yp/data/cookie_theft/Manchester/lists/binary_list' 
data_list = open(data_list_path).readlines()

for idx in os.listdir(list_folder):
    list_path = os.path.join(list_folder, idx, 'train_list')
    
    list_f = open(list_path, 'a')
    for line in data_list:
        list_f.write(line)
    list_f.close()
    
    
    
    