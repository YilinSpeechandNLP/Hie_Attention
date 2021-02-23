import numpy as np
import os
import pdb
txt_folder = '/mnt/fastdata/ac1yp/iva_2020/cookie_theft_txt/'
wav_folder = '/mnt/fastdata/ac1yp/iva_2020/wav_cookie_theft/'

for file_id in os.listdir(wav_folder):
    rec_id = file_id.split('.')[0]
    if os.path.exists(os.path.join(txt_folder, rec_id)) == False:
        print(rec_id)
    