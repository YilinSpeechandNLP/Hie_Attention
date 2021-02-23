import os
import numpy as np
import pdb
# 0:HC, 1:FMD, 2:MCI, 3:ND 
'''
file_path = '/data/ac1yp/data/cookie_theft/IVA-all/label_info-4ways'
cd_folder = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_txt'
lst_path = '/data/ac1yp/data/cookie_theft/IVA-all/all_list'
file_content = open(file_path).readlines()
label_dict = {}
HC_idx = 0
FMD_idx = 0
MCI_idx = 0
ND_idx = 0
for line in file_content:
    line = line.split('\t')
    rec_id = line[0][3:]
    label = line[1][:-1]
    print(rec_id)
    if label == 'HC':
        l_id = 0
        HC_idx +=1
    elif label == 'FMD':
        l_id = 1
        FMD_idx +=1
    elif label == 'MCI':
        l_id = 2
        MCI_idx +=1
    elif label == 'ND':
        l_id = 3
        ND_idx +=1
    else:
        continue
    label_dict[rec_id] = l_id
    
    
print(label_dict)
print(HC_idx, FMD_idx, MCI_idx, ND_idx)
lst_f = open(lst_path,'w')
new_label_dict = {}
for rec_id in os.listdir(cd_folder):
    if rec_id not in label_dict.keys():
        print(rec_id)
    else:
        lst_f.write(rec_id+'\n')
        new_label_dict[rec_id] = label_dict[rec_id]
lst_f.close()
np.save('/data/ac1yp/data/cookie_theft/IVA-all/label_dict.npy', new_label_dict)

HC_idx = 0
FMD_idx = 0
MCI_idx = 0
ND_idx = 0
label_dict = np.load('/data/ac1yp/data/cookie_theft/IVA-all/label_dict.npy', allow_pickle=True).item()
print(len(label_dict.keys()))
for rec_id in label_dict.keys():
    label = label_dict[rec_id]
    if label == 0:
        HC_idx +=1
    elif label == 1:
        FMD_idx +=1
    elif label == 2:
        MCI_idx +=1
    else:
        ND_idx +=1
print(HC_idx, FMD_idx, MCI_idx, ND_idx)

label_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/label_dict-cookie_theft.npy'
list_path = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/all_list'
output_path = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/2-ways-lists/label_dict'
label_dict = np.load(label_dict_path, allow_pickle=True).item()
list_cont = open(list_path).readlines()
sub_label_dict = {}


for rec_id in list_cont:
    label=label_dict[rec_id[:-1]]
    if label == 0 or label == 1:
        sub_label_dict[rec_id[:-1]] = 0
    elif label == 3 or label == 2:
        sub_label_dict[rec_id[:-1]] = 1
np.save(output_path, sub_label_dict)
print(len(sub_label_dict.keys()))
print(sub_label_dict)


## bristol

input_list_path = '/data/ac1yp/data/cookie_theft/bristol/diagnosis_type'
wav_folder = '/data/ac1yp/data/cookie_theft/bristol/audio'
output_dict_path = '/data/ac1yp/data/cookie_theft/bristol/3class_dict'
input_list = open(input_list_path).readlines()
label_dict = {}
for line in input_list:
    rec_id = line.split('\t')[0]
    label = line.split('\t')[1][:-1]

    if label == 'normal':
        label_id = 0
    elif label == 'mci':
        label_id = 1
    elif label == 'dementia' :
        label_id = 2
    else:
        continue
    if os.path.exists(os.path.join(wav_folder, rec_id+'_cook.wav')) == True:
        #pdb.set_trace()
        label_dict[rec_id+'_cook'] = label_id
print(len(label_dict.keys()))
np.save(output_dict_path, label_dict)
''' 

DB_dict_path = '/data/ac1yp/data/cookie_theft/DementiaBank/label_dict.npy'
ADReSS_dict_path = '/data/ac1yp/data/cookie_theft/ADReSS/ADReSS_label_dict.npy'
Bristol_dict_path = '/data/ac1yp/data/cookie_theft/bristol/HC_ND_dict.npy'
IVA_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/2-ways-lists/label_dict.npy'
Man_dict_path = '/data/ac1yp/data/cookie_theft/Manchester/label_dict.npy'

output_path = '/data/ac1yp/data/cookie_theft/label_dict'

path_list = [DB_dict_path, ADReSS_dict_path, Bristol_dict_path, IVA_dict_path, Man_dict_path]

comb_dict = {}
for path in path_list:
    sub_dict = np.load(path, allow_pickle=True).item()
    print(sub_dict.keys())
    comb_dict.update(sub_dict)
    
np.save(output_path, comb_dict)
print(len(comb_dict.keys()))    
