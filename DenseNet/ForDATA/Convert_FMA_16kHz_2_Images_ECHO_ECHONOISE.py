#===============================================================================
# Convert_FMA_16kHz_2_ImagesECHO_ECHONOISE.py
# CONVERT DATA TO IMAGE
#  Echoed files and Echoed files adding noise -> 2 X DATA
# Inputs:
#   Echoed Wave files sr = 16 kHz, PCM 16 bits
# Outputs:
#    Mel Spectrogram Images for Echoed files and Echoed files adding noise
# Authors: DTLT, TVL
#=================================================================================

from PIL import Image
import numpy as np
import sys
import librosa
import os


ltime_series = 230          # Number of frames
parameter_number = 230      # Number of Mel Coeff.
total_file = 8000           # 1000 x 8 genres
NFFT = 2*2048
hop_length = NFFT // 2
genres = 'Electronic Experimental Folk Hip_Hop Instrumental International Pop Rock'.split()

fs = 16000
frequency_max = fs//2

# Init array containing data for image
data = np.zeros((ltime_series, parameter_number), dtype=np.float32)

# Directory containing images
image_dir = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
i = 0                       # Number of sound files
for g in genres:
    for k, filename in enumerate(os.listdir(f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm_echo/{g}')):
        songname = f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm_echo/{g}/{filename}'
        
        y, sr = librosa.load(songname, mono=True, sr=fs)  # Downsampling to 16 kHz
        duration = librosa.get_duration(y=y, sr=sr)  # 오디오 길이 계산
        
        # ADD: 29.35초 이상인 파일만 선택
        if duration < 29.35:
            continue  # 29.35초 미만 파일은 건너뜀
        
        print(songname.ljust(85), "  i =  ", i, "/", total_file)
        filename3 = filename.split("_")[0] + "_PCM16_ECHO.wav"
        songname3= f'/home/hwang-gyuhan/Workspace/dataset/Spotify_wav_pcm_echo/{g}/{filename3}'
        file_name3 = image_dir + filename3.split("_")[0] +"_ECHO.png"
        file_name4 = image_dir + filename3.split("_")[0] +"_ECHO_NOISE.png"
        print ("songname3 = ", songname3)
        print ("file_name3 = ", file_name3)
        print ("file_name4 = ", file_name4)

        # ECHO files------------------------------------------------------
        y3, sr = librosa.load(songname3, mono=True,
                              sr=fs) # now fs = 16 kHz,
        S3 = librosa.feature.melspectrogram(y=y3, sr=fs,
                                            n_mels=parameter_number,
                                            n_fft=NFFT,
                                            hop_length=NFFT // 2,
                                            win_length=NFFT,
                                            window='hamm',
                                            center=True,
                                            pad_mode='reflect', power=2.0, fmax=frequency_max
                                            )

        S_dB3 = librosa.power_to_db(S3, ref=np.max)
        data[:, 0:parameter_number]= -S_dB3.T[0:ltime_series, :]
        data_max = np.max(data)
        img = Image.fromarray(np.uint8((data / data_max) * 255), 'L')
        img.save(file_name3)


        # Adding noise to ECHO files---------------------------------------
        noise_amp3 = 0.03 * np.random.uniform(size=len(y3)) * np.amax(y3)
        y3noise = y3 + noise_amp3

        S4 = librosa.feature.melspectrogram(y=y3noise, sr=fs,
                                            n_mels=parameter_number,
                                            n_fft=NFFT,
                                            hop_length=NFFT // 2,
                                            win_length=NFFT,
                                            window='hamm',
                                            center=True,
                                            pad_mode='reflect', power=2.0, fmax=frequency_max
                                            )

        S_dB4 = librosa.power_to_db(S4, ref=np.max)
        data [:, 0:parameter_number]= -S_dB4.T[0:ltime_series, :]
        data_max = np.max(data)
        img = Image.fromarray(np.uint8((data / data_max) * 255), 'L')
        img.save(file_name4)
        i=i+1

