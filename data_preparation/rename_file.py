import os
import numpy as np
import shutil 
'''
text_folder = '/data/ac1yp/data/cookie_theft/IVA-all/original_text/'
output_folder = '/data/ac1yp/data/cookie_theft/IVA-all/selected_text/'
list_path = '/data/ac1yp/data/cookie_theft/IVA-all/cookie_theft_info/all_list'
list_content = open(list_path).readlines()

for file_name in os.listdir(text_folder):
    file_path = os.path.join(text_folder, file_name)
    if len(file_name.split('.')[0]) == 3:
        file_name = '0'+file_name
    output_path = os.path.join(text_folder, file_name)
    os.rename(file_path, output_path)

for line in list_content:
    text_path = os.path.join(text_folder, line[:-1]+'.txt')
    output_path = os.path.join(output_folder, line[:-1]+'.txt')
    if os.path.exists(text_path) == False:
        print(line[:-1])
    else:
        shutil.copyfile(text_path, output_path)
'''

audio_folder = '/data/ac1yp/data/cookie_theft/Manchester/audio/'
transcript_folder = '/data/ac1yp/data/cookie_theft/Manchester/transcript/'

for org_name in os.listdir(transcript_folder):
    new_name = '-'.join(org_name.split('_'))
    print(new_name)
    os.rename(os.path.join(transcript_folder, org_name), os.path.join(transcript_folder, new_name))
    