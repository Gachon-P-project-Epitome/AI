import os
import librosa
import numpy as np

def segment_audio_files(folder_path, segment_duration=2.9, output_folder='output_segments'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 출력 폴더 생성

    total_files_processed = 0  # 처리된 파일 수 초기화
    excluded_files = {'098569.mp3', '098567.mp3', '098565.mp3'}  # 제외할 파일 이름 집합

    # 지정된 폴더 내의 모든 오디오 파일 처리
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.mp3'):  # MP3 파일 처리
                # 파일 이름에서 확장자를 제거하고 확인
                file_base_name = os.path.splitext(filename)[0]
                if file_base_name in excluded_files:
                    print(f"제외된 파일: {filename}")  # 제외된 파일 알림
                    continue

                total_files_processed += 1  # 파일 처리 수 증가
                
                # MP3 파일의 전체 경로
                file_path = os.path.join(dirpath, filename)

                print(f"처리 중: {filename}")  # 파일 처리 시작 알림
                
                # 오디오 파일 로드 (sr=None으로 설정하여 원본 샘플링 주파수 사용)
                try:
                    audio, sr = librosa.load(file_path, sr=44100, duration=29.0)  # 29초까지 로드
                except Exception as e:
                    print(f"파일 로드 오류: {filename} - {str(e)}")
                    continue
                
                # 총 샘플 수와 세그먼트의 샘플 수 계산
                total_samples = len(audio)
                segment_samples = int(segment_duration * sr)  # 2.9초에 해당하는 샘플 수
                segment_count = total_samples // segment_samples  # 가능한 세그먼트 수
                
                print(f"파일: {filename}, 총 길이(샘플): {total_samples}, 세그먼트 수: {segment_count}")

                # 세그먼트가 생성될 수 있는지 확인
                if segment_count == 0:
                    print(f"  -> {filename}: 세그먼트를 만들 수 없습니다. (총 샘플 수: {total_samples})")
                    continue

                # 세그먼트를 저장할 리스트
                segments = []

                # 세그먼트 나누기
                for i in range(segment_count):
                    start_sample = i * segment_samples
                    end_sample = start_sample + segment_samples
                    
                    segment = audio[start_sample:end_sample]
                    segments.append(segment)  # 세그먼트를 리스트에 추가

                    # 세그먼트 출력 (진폭값)
                    print(f"  -> 세그먼트 {i + 1} 저장됨: {segment[:10]}...")  # 앞 10개 샘플만 출력

                # 모든 세그먼트를 하나의 NPZ 파일로 저장
                if segments:
                    segments_array = np.array(segments, dtype=np.float32)  # 리스트를 배열로 변환하며 데이터 타입 지정
                    output_filename = f"{os.path.splitext(filename)[0]}_all_segments.npz"
                    output_path = os.path.join(output_folder, output_filename)
                    np.savez(output_path, segments=segments_array)  # 세그먼트를 npz 형식으로 저장
                    print(f"{output_filename} 저장 완료.\n")  # 파일 저장 완료 알림

    print(f"총 처리된 MP3 파일 수: {total_files_processed}")  # 처리된 파일 수 출력

# 사용 예시
folder_path = '/home/hwang-gyuhan/Workspace/Tabnet/fma_small'  # MP3 파일이 있는 상위 폴더 경로
output_folder = '/home/hwang-gyuhan/Workspace/Transformer/output_segments'  # 세그먼트를 저장할 폴더 경로
segment_audio_files(folder_path, output_folder=output_folder)