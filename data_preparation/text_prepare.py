import numpy as np
from scipy.io import wavfile


fs,signal = wavfile.read(os.path.join(data_folder,wav_name))