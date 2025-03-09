import librosa
import matplotlib.pyplot as plt
import numpy as np

def analyze_frequency(mp3_path1, mp3_path2, output_path="frequency_analysis.png"):
    """
    두 MP3 파일의 주파수를 분석하고 그래프로 저장합니다.
    :param mp3_path1: 첫 번째 MP3 파일 경로
    :param mp3_path2: 두 번째 MP3 파일 경로
    :param output_path: 저장할 그래프 파일 경로
    """
    # 첫 번째 MP3 파일 로드
    y1, sr1 = librosa.load(mp3_path1, sr=None)  # 원래 샘플링 레이트 유지
    
    # 두 번째 MP3 파일 로드
    y2, sr2 = librosa.load(mp3_path2, sr=None)

    # Short-Time Fourier Transform (STFT)으로 주파수 분석
    D1 = librosa.stft(y1)
    D2 = librosa.stft(y2)

    # 주파수 크기 스펙트럼 계산
    S1 = librosa.amplitude_to_db(abs(D1), ref=np.max)
    S2 = librosa.amplitude_to_db(abs(D2), ref=np.max)

    # 그래프 생성
    plt.figure(figsize=(10, 8))

    # 첫 번째 MP3 주파수 스펙트럼
    plt.subplot(2, 1, 1)
    time1 = np.linspace(0, len(y1) / sr1, S1.shape[1])  # 시간 축 계산
    freq1 = np.linspace(0, sr1 / 2, S1.shape[0])        # 주파수 축 계산
    plt.pcolormesh(time1, freq1, S1, shading='gouraud', cmap='viridis')  # 주파수 스펙트럼 플롯
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Frequency Spectrum of {mp3_path1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 두 번째 MP3 주파수 스펙트럼
    plt.subplot(2, 1, 2)
    time2 = np.linspace(0, len(y2) / sr2, S2.shape[1])  # 시간 축 계산
    freq2 = np.linspace(0, sr2 / 2, S2.shape[0])        # 주파수 축 계산
    plt.pcolormesh(time2, freq2, S2, shading='gouraud', cmap='viridis')  # 주파수 스펙트럼 플롯
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Frequency Spectrum of {mp3_path2}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 그래프 저장
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # 그래프를 파일로 저장
    print(f"그래프가 저장되었습니다: {output_path}")

# 사용 예제
mp3_file1 = '/home/hwang-gyuhan/Workspace/dataset/Spotify/Electronic/Electronic5KcKUpTEHMfcoAps9d5BvY.mp3'  # 첫 번째 MP3 파일 경로
mp3_file2 = '/home/hwang-gyuhan/Workspace/testreal.mp3'  # 두 번째 MP3 파일 경로
analyze_frequency(mp3_file1, mp3_file2, output_path="/home/hwang-gyuhan/Workspace/frequency_real.png")