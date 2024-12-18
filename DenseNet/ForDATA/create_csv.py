import os
import random
import pandas as pd

# fma_wav_pcm 폴더 경로
audio_dir = '/home/hwang-gyuhan/Workspace/dataset/fma_wav_pcm/'

# FMA_IMAGES 폴더 경로
image_dir = '/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/'

# CSV 파일을 저장할 경로
train_valid_csv = '/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/train_valid_data.csv'
test_csv = '/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/test_data.csv'

# 장르 정보를 담을 딕셔너리
genre_dict = {}

# fma_wav_pcm 폴더에서 장르와 오디오 파일을 추출
for genre_name in os.listdir(audio_dir):
    genre_path = os.path.join(audio_dir, genre_name)
    if os.path.isdir(genre_path):
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.wav'):
                song_id = audio_file.split('_')[0]  # 예시: 001482_PCM16.wav -> 001482
                genre_dict[song_id] = genre_name  # song_id와 장르를 매핑

# 오디오 트랙에 대한 이미지 파일 목록을 추출합니다
audio_tracks = {}

# 이미지 파일 읽기
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        # song_id를 filename에서 추출 (예: 000002.png -> 000002)
        song_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
        
        # song_id에 해당하는 이미지 그룹을 dictionary에 추가
        if song_id not in audio_tracks:
            audio_tracks[song_id] = []
        audio_tracks[song_id].append(filename)

# 데이터셋을 train+valid와 test로 나누기 (9:1 비율)
train_valid_data = []
test_data = []

# 각 오디오 트랙의 이미지들을 train+valid와 test로 나눔
for song_id, images in audio_tracks.items():
    if song_id in genre_dict:  # 장르 정보가 있는 경우만 처리
        genre_name = genre_dict[song_id]  # 해당 song_id에 대한 장르
        
        # 데이터 나누기
        if random.random() < 0.9:
            # train+valid에 추가
            for img in images:
                train_valid_data.append([img, f"['{genre_name}']"])  # 장르를 리스트로 감싸서 저장
        else:
            # test에 추가
            for img in images:
                test_data.append([img, f"['{genre_name}']"])  # 장르를 리스트로 감싸서 저장

# train+valid 데이터 CSV 생성
train_valid_df = pd.DataFrame(train_valid_data, columns=['song_id', 'genre_name'])
train_valid_df.to_csv(train_valid_csv, index=False)

# test 데이터 CSV 생성
test_df = pd.DataFrame(test_data, columns=['song_id', 'genre_name'])
test_df.to_csv(test_csv, index=False)

print(f"train+valid 데이터 CSV 파일 생성 완료: {train_valid_csv}")
print(f"test 데이터 CSV 파일 생성 완료: {test_csv}")