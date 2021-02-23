import os
import numpy as np
from scipy.io import wavfile
'''
iva_question_path = '/mnt/fastdata/ac1yp/iva_2020/iva_questions.txt'
iva_question_content = open(iva_question_path).readlines()

cookie_theft_question_time_path ='/mnt/fastdata/ac1yp/iva_2020/cookie_theft_questions.txt'
ct_f = open(cookie_theft_question_time_path, 'w')
cd_dict = {}
for line in iva_question_content:
    question_id = line.split(' ')[2]
    if question_id == 'cd':
        name = line.split(' ')[0]
        if len(name) == 3:
            new_name = '0'+name
        else:
            new_name = name
        start_t = line.split(' ')[4]
        end_t = line.split(' ')[5][:-1]
        cd_dict[new_name] = [float(start_t), float(end_t)]
        ct_f.write(line)
ct_f.close()
print(cd_dict)
np.save('/mnt/fastdata/ac1yp/iva_2020/cookie_theft_questions', cd_dict)
'''
cd_dict = np.load('/mnt/fastdata/ac1yp/iva_2020/cookie_theft_questions.npy', allow_pickle=True).item()
wav_folder = '/mnt/fastdata/ac1yp/iva_2020/wav'
output_folder = '/mnt/fastdata/ac1yp/iva_2020/wav_cookie_theft'
for rec_id in cd_dict.keys():
    wav_path = os.path.join(wav_folder, rec_id+'.wav')
    fs, signal = wavfile.read(wav_path)
    start_t, end_t = cd_dict[rec_id]
    start_fs = int(start_t*fs)
    end_fs = int(end_t*fs)
    
    cd_signal = signal[start_fs:end_fs]
    output_path = os.path.join(output_folder, rec_id+'.wav')
    wavfile.write(output_path, fs, cd_signal)
    print(rec_id)