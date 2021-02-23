import os

wav_folder = '/data/ac1yp/data/cookie_theft/Manchester/audio/'
trans_folder = '/data/ac1yp/data/cookie_theft/Manchester/transcript/'

for file in os.listdir(wav_folder):
    if os.path.exists(os.path.join(trans_folder,file.split('.')[0])) == False:
        print(file)