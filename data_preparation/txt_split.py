import numpy as np
import os
import pdb
import math
import shutil
'''
input_text_path = '/mnt/fastdata/ac1yp/iva_2020/text'
output_txt_path = '/mnt/fastdata/ac1yp/iva_2020/cookie_theft_txt'
input_text = open(input_text_path).readlines()

align_info_path = '/mnt/fastdata/ac1yp/iva_2020/cookie_theft_questions.npy'
align_info_dict = np.load(align_info_path, allow_pickle=True).item()
idx = 0
rec_dict = {}
for line in input_text:
    rec_id = line.split('-')[0][3:]
    time_info = line.split(' ')[0].split('_')[-1]
    spk_id = line.split('-')[1][0]
    if rec_id in align_info_dict and spk_id == 'B':
        ### have cookie theft description 
        start_t, end_t = align_info_dict[rec_id]
        start_int=math.floor(start_t*100)
        end_int = math.ceil(end_t*100)
        t_1 = time_info.split('-')[0]
        t_2 = time_info.split('-')[1]
        t_1 = int(t_1)
        t_2 = int(t_2)
        
        if rec_id == '0282':
            pdb.set_trace()
        #print(t_1, start_int, t_2, end_int)
        if t_1 >= start_int and t_2 <= end_int:
            content = line.split(' ')[1:]
            content = ' '.join(content)[:-1]+'.'
            content = content.lower()
            if rec_id in rec_dict.keys():
                
                rec_dict[rec_id].append(content)
            else:
                rec_dict[rec_id] = [content]

for rec_id in rec_dict.keys():
    content = rec_dict[rec_id]
    content = ' '.join(content)
    output_path = os.path.join(output_txt_path, rec_id)
    output_f = open(output_path,'w')
    output_f.write(content+'\n')
    output_f.close()
'''          
input_txt_folder = '/mnt/fastdata/ac1yp/iva_2020/cookie_theft_txt'

for file_id in os.listdir(input_txt_folder):
    input_folder = os.path.join(input_txt_folder, file_id)
    content = open(input_folder).readlines()[0]
    
    content = content.lower()
    f_=open(input_folder,'w')
    f_.write(content)
    f_.close()
    

        