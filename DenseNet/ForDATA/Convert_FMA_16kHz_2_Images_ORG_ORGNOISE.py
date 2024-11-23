#===============================================================================
# Convert_FMA_16kHz_ImageX2.py
# CONVERT DATA TO IMAGE
# Original files and Original files adding noise -> 2 X DATA
# Inputs:
#   Wave files sr = 16 kHz, PCM 16 bits
# Outputs:
#    Mel Spectrogram Images for Original files and Original files adding noise
# Authors: DTLT, TVL
#=================================================================================

from PIL import Image
import numpy as np
import sys
import librosa
import os

ltime_series = 230      # Number of frames x Time of Frame = 30 s
parameter_number = 230  # Number of Mel Coeff.
total_file = 8000       # 1000 x 8 genres
NFFT = 2*2048
hop_length = NFFT // 2
fs = 16000
frequency_max = fs//2
genres = 'Electronic Experimental Folk Hip_Hop Instrumental International Pop Rock'.split()

# Init array containing data for image
data = np.zeros((ltime_series, parameter_number), dtype=np.float32)

# Directory containing images
image_dir = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
i = 0                   # Number of sound files
for g in genres:
    for k, filename in enumerate(os.listdir(f'/home/hwang-gyuhan/Workspace/dataset/fma_wav_pcm/{g}')):
        songname = f'/home/hwang-gyuhan/Workspace/dataset/fma_wav_pcm/{g}/{filename}'
        print(songname.ljust(85), "  i =  ", i, "/", total_file)
        print(filename)
        filename1 = image_dir + filename.split("_")[0] +".png"
        filename2 = image_dir + filename.split("_")[0] +"_Noise.png"
        print ("filename1 = ",filename1)
        print ("filename2 = ", filename2)
        y, sr = librosa.load(songname, mono=True, sr=fs)  # Downsampling to 16 kHz
        print ("Sampling Frequency = ",sr)
        noise_amp1 = 0.03 * np.random.uniform(size=len(y)) * np.amax(y)
        ynoise = y + noise_amp1
        # Nothing change -------------------------------------------------------
        S = librosa.feature.melspectrogram(y=y, sr=fs,
                                            n_mels=parameter_number,
                                            n_fft=NFFT,
                                            hop_length=NFFT // 2,
                                            win_length=NFFT,
                                            window='hamm',
                                            center=True,
                                            pad_mode='reflect', power=2.0,
                                            fmax = frequency_max
                                            )

        S_dB = librosa.power_to_db(S, ref=np.max)
        data[:, 0:parameter_number]= -S_dB.T[0:ltime_series, :]
        data_max = np.max(data)
        img = Image.fromarray(np.uint8((data / data_max) * 255), 'L')
        img.save(filename1)
        # Adding noise-----------------------------------------------------------------
        Snoise = librosa.feature.melspectrogram(y=ynoise, sr=fs,
                                            n_mels=parameter_number,
                                            n_fft=NFFT,
                                            hop_length=NFFT // 2,
                                            win_length=NFFT,
                                            window='hamm',
                                            center=True,
                                            pad_mode='reflect', power=2.0, fmax=frequency_max
                                            )

        S_dBnoise = librosa.power_to_db(Snoise, ref=np.max)
        data[:, 0:parameter_number] = -S_dBnoise.T[0:ltime_series, :]
        data_max = np.max(data)
        img = Image.fromarray(np.uint8((data/ data_max) * 255), 'L')
        img.save(filename2)
        i=i+1
