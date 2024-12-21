from PIL import Image
import numpy as np
import librosa
import os
from Preprocessing import *
import sys
sys.path.append('/home/hwang-gyuhan/Workspace/DenseNet/TestCode')


class FeatureExtraction:
    
    def __init__(self, ltime_series=230,  # Number of frames x Time of Frame = 30 s
                 parameter_number=230,    # Number of Mel Coeff.
                 total_file=1,
                 NFFT= 2*2048, 
                 fs=16000):
        
        self.ltime_series = ltime_series
        self.parameter_number = parameter_number
        self.total_file = total_file
        self.NFFT = NFFT
        self.hop_length = self.NFFT // 2
        self.fs = fs
        self.frequency_max = self.fs // 2
        
    def mel_spectrogram(self, pcm_data):
        # PCM 데이터를 바로 처리
        with open(pcm_data, 'rb') as pcm_file:
            pcm_file.seek(0)  # PCM 데이터 스트림의 시작으로 이동
            y, sr = librosa.load(pcm_file, sr=self.fs, mono=True)
        
        # Init array containing data for image
        data = np.zeros((self.ltime_series, self.parameter_number), dtype=np.float32)
        duration = librosa.get_duration(y=y, sr=sr)  # 오디오 길이 계산
        
        if duration < 29.35:
            return None  
        
        # Nothing change -------------------------------------------------------
        S = librosa.feature.melspectrogram(y=y, sr=self.fs,
                                            n_mels=self.parameter_number,
                                            n_fft=self.NFFT,
                                            hop_length=self.NFFT // 2,
                                            win_length=self.NFFT,
                                            window='hamm',
                                            center=True,
                                            pad_mode='reflect', power=2.0,
                                            fmax = self.frequency_max
                                            )

        S_dB = librosa.power_to_db(S, ref=np.max)
        data[:, 0:self.parameter_number]= -S_dB.T[0:self.ltime_series, :]
        data_max = np.max(data)
        img = Image.fromarray(np.uint8((data / data_max) * 255), 'L')
    
        return img