import os
import numpy as np 


mmse_info_path = '/data/ac1yp/data/cookie_theft/IVA-all/mmse_info'
mmse_dict_path = '/data/ac1yp/data/cookie_theft/IVA-all/mmse_dict'
mmse_info = open(mmse_info_path).readlines()
mmse_dict = {}
for line in mmse_info:
    line = line.split('\t')
    rec_id = line[0]
    mmse_val = line[1]
    if len(rec_id) == 1:
        rec_id = '020'+rec_id
    elif len(rec_id) == 2:
        rec_id = '02'+rec_id
    elif len(rec_id) == 3:
        rec_id = '2'+rec_id
    if mmse_val == '\n':
        print(rec_id)
    else:
        mmse_dict[rec_id] = int(mmse_val[:-1])
np.save(mmse_dict_path, mmse_dict)