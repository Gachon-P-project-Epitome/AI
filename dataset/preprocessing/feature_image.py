import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 상위 폴더 경로 설정
base_folder = '/home/hwang-gyuhan/Workspace/Tabnet/fma_small/'  # 여기에 상위 폴더 경로를 입력하세요
output_folder = '/home/hwang-gyuhan/Workspace/dataset/feature_image_all/chroma_cens/'  # 저장할 경로를 여기에 직접 지정

# 출력 폴더가 없다면 생성
os.makedirs(output_folder, exist_ok=True)

# MP3 파일 개수 카운트
mp3_count = 0
max_mp3_count = 8000  # 최대 MP3 파일 개수

# 모든 서브 폴더 및 파일을 순회
for folder in range(156):  # 000부터 155까지
    folder_path = os.path.join(base_folder, f'{folder:03d}')  # 폴더 경로 생성
    if os.path.isdir(folder_path):  # 해당 경로가 폴더인지 확인
        for file in os.listdir(folder_path):
            if file.endswith('.mp3'):
                audio_file_path = os.path.join(folder_path, file)

                try:
                    # 오디오 파일 로드
                    y, sr = librosa.load(audio_file_path)

                    # Chroma CENS feature 추출
                    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)

                    # Chroma CENS 시각화
                    plt.figure(figsize=(12, 6))
                    img = librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time', sr=sr)
                    plt.title(f'Chroma CENS - {file}')
                    plt.colorbar(label='Energy')

                    # 파일 이름에서 확장자를 제거하고 저장할 파일 경로 생성
                    output_file_path = os.path.join(output_folder, f'{os.path.splitext(file)[0]}_chroma_cens.png')

                    # 그래프 이미지를 PNG 파일로 저장
                    plt.savefig(output_file_path)

                    # 그래프 닫기
                    plt.close()

                    # MP3 파일 카운트 증가
                    mp3_count += 1

                    # 최대 MP3 파일 수에 도달하면 종료
                    if mp3_count >= max_mp3_count:
                        break

                except Exception as e:
                    print(f"Error processing {audio_file_path}: {e}")

    if mp3_count >= max_mp3_count:
        break