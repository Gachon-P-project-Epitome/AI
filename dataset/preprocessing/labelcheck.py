import numpy as np

# npz 파일에서 데이터 로드 및 첫 5줄 출력
def load_and_print_npz(npz_file_path):
    data = np.load(npz_file_path)
    
    # track_id와 genre 배열 추출
    track_ids = data['track_id']
    genres = data['genre']
    
    print("First 5 rows from the npz file:")
    for i in range(10):
        print(f"Track ID: {track_ids[i]}, One-Hot Encoded Genre: {genres[i]}")

# npz 파일 경로 설정
npz_file_path = '/home/hwang-gyuhan/Workspace/Tabnet/track_genre_data.npz'

# 첫 5줄 출력 함수 실행
load_and_print_npz(npz_file_path)