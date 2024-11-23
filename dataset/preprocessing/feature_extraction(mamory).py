import warnings
import os
import multiprocessing
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm



# 피처를 추출할 폴더 경로를 지정
AUDIO_DIR = '/home/hwang-gyuhan/Workspace/Tabnet/fma_small'

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rms=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1,
                         spectral_flatness=1)  
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    return columns

def compute_features(filepath):
    tid = os.path.basename(filepath)
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        x, sr = librosa.load(filepath, sr=None, mono=True)

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        if np.isnan(stft).any() or np.isinf(stft).any():
            print(f"STFT contains NaN or inf values for {filepath}")
            return tid, features
        
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)
        f = librosa.feature.rms(S=stft)
        feature_stats('rms', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)
        f = librosa.feature.spectral_flatness(S=stft)
        feature_stats('spectral_flatness', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        if np.isnan(mel).any() or np.isinf(mel).any():
            print(f"Mel spectrogram contains NaN or inf values for {filepath}")
            return tid, features
    
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print('{}: {}'.format(filepath, repr(e)))

    # Pandas Series 객체를 NumPy 배열로 변환하여 반환
    return tid, features.values

def main():
    # MP3 파일 경로를 찾습니다.
    mp3_files = sorted([os.path.join(root, file)
                        for root, dirs, files in os.walk(AUDIO_DIR)
                        for file in files if file.endswith('.mp3')])

    features_list = []  # 피처 배열을 저장할 리스트

    # Get the number of CPU cores
    nb_workers = os.cpu_count()

    print(f'Working with {nb_workers} processes.')

    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(compute_features, mp3_files)

    for features in tqdm(it, total=len(mp3_files)):
        features_list.append(features[1])  # features[1]는 피처 배열

    # features_list는 모델의 입력으로 사용할 수 있는 NumPy 배열로 변환
    features_array = np.array(features_list)

    return features_array  # 피처 배열 반환

if __name__ == "__main__":
    features = main()  # 피처 배열 호출
