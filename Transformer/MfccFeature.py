import os
import numpy as np
import librosa

def extract_mfcc_from_npz(npz_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 출력 폴더 생성

    total_files_processed = 0  # 처리된 파일 수 초기화

    # 지정된 폴더 내의 모든 NPZ 파일 처리
    for filename in os.listdir(npz_folder):
        if filename.endswith('.npz'):
            total_files_processed += 1  # 파일 처리 수 증가
            npz_path = os.path.join(npz_folder, filename)

            print(f"처리 중: {filename}")  # 파일 처리 시작 알림
            
            # NPZ 파일에서 세그먼트 배열 로드
            with np.load(npz_path) as data:
                segments = data['segments']  # 'segments' 키로 세그먼트 배열 불러오기

            # MFCC 추출을 위한 세그먼트 리스트
            mfcc_segments = []

            # 각 세그먼트에 대해 MFCC 추출
            for segment in segments:
                mfcc = librosa.feature.mfcc(y=segment, sr=44100, n_mfcc=20)  # MFCC 추출
                mfcc_segments.append(mfcc)  # MFCC 배열을 리스트에 추가

            # MFCC 배열을 NumPy 배열로 변환
            mfcc_segments_array = np.array(mfcc_segments)  # (10, x) 형태로 변환
            
            # MFCC 결과를 NPZ 파일로 저장
            output_filename = f"{os.path.splitext(filename)[0]}_mfcc.npz"
            output_path = os.path.join(output_folder, output_filename)
            np.savez(output_path, mfcc=mfcc_segments_array)  # MFCC 세그먼트를 npz 형식으로 저장
            print(f"{output_filename} 저장 완료.\n")  # 파일 저장 완료 알림

    print(f"총 처리된 NPZ 파일 수: {total_files_processed}")  # 처리된 파일 수 출력

# 사용 예시
npz_folder = '/home/hwang-gyuhan/Workspace/Transformer/output_segments'  # NPZ 파일이 있는 폴더 경로
output_folder = '/home/hwang-gyuhan/Workspace/Transformer/mfcc_segments'  # MFCC를 저장할 폴더 경로
extract_mfcc_from_npz(npz_folder, output_folder)