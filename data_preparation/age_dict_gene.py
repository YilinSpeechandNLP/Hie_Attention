import os
import numpy as np
import pdb
'''
trs_path = '/mnt/fastdata/ac1yp/iva_2020/age_gender'

output_age_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/age_dict'
output_gender_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/gender_dict'
file_cont = open(trs_path).readlines()
age_dict = {}
gender_dict = {}
for line in file_cont:
    line = line.split('\t')
    rec_id = line[0]
    if len(rec_id) == 1:
        rec_id = '020'+rec_id
    elif len(rec_id) == 2:
        rec_id = '02'+rec_id
    elif len(rec_id) == 3:
        rec_id = '2'+rec_id    
    if line[1] != 'UNK':
        age = int(line[1])
    else:
        continues
    gender = line[2][:-1]
    gender_dict[rec_id] = gender
    age_dict[rec_id] = age
np.save(output_age_dict_path,age_dict)
np.save(output_gender_dict_path,gender_dict)
print(age_dict)
print(gender_dict)
'''
age_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/mmse_dict.npy'
age_dict = np.load(age_dict_path, allow_pickle=True).item()

label_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/label_dict-cookie_theft.npy'
label_dict = np.load(label_dict_path, allow_pickle=True).item()



new_age_dict = {}
for rec_id in label_dict.keys():
    if rec_id in age_dict.keys():
        age = age_dict[rec_id]
        new_age_dict[rec_id] = age
    else:
        print(rec_id)
print(len(new_age_dict.keys()))
np.save('/data/ac1yp/data/cookie_theft/IVA-all/mmse_dict-cookie_theft.npy', new_age_dict)
