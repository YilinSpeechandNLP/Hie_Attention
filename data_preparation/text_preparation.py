import os
import numpy as np
import pdb
import re
ADReSS_trans_folder = '/data/ac1yp/data/cookie_theft/Manchester/transcript/'
text_path = '/fastdata/ac1yp/ASR/TL_cookie_theft/Manchester/text'
refer_file_path = '/fastdata/ac1yp/ASR/TL_cookie_theft/Manchester/utt2dur'

text_f = open(text_path,'w')
refer_file = open(refer_file_path).readlines()

def text_regular(text):
  punctuation = '!,;:?".+//<>'
  text = re.sub(r'[{}]+'.format(punctuation),'',text)
  text = re.sub(' +', ' ', text)
  text = re.sub('xxx', '<UNK>', text)
  return text.strip().upper()

for line in refer_file:
    spk_id=line.split(' ')[0]
    t_path = os.path.join(ADReSS_trans_folder, spk_id)
    if os.path.exists(t_path) == False:
        print(spk_id)
        continue
    text = open(t_path, encoding='ascii', errors='ignore').readlines()[0]
    text=text_regular(text)
    text_f.write(spk_id+' '+text+'\n')
text_f.close()
