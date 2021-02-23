import numpy as np

'''
split_wav='/data/ac1yp/data/cookie_theft/ADReSS/split_audio/'
full_wav='/data/ac1yp/data/cookie_theft/ADReSS/audio/'
silence_wav_path = '/data/ac1yp/data/cookie_theft/ADReSS/silence.wav'
spk2utt_path = '/fastdata/ac1yp/ASR/TL_cookie_theft/ADReSS/spk2utt'

spk2utt = open(spk2utt_path).readlines()
output_path= '/data/ac1yp/data/cookie_theft/ADReSS/wav_concatenate.sh'
output_f = open(output_path,'w')
output_f.write('#!/bin/bash\n')
for line in spk2utt:
    line = line.split(' ')
    rec_id=line[0]
    output_f.write('sox ')
    for wav_id in line[1:-1]:
        output_f.write(split_wav+wav_id+'.wav ')
    output_f.write(split_wav+line[-1][:-1]+'.wav ')
    output_f.write(full_wav+rec_id+'.wav\n')
output_f.close()
'''


