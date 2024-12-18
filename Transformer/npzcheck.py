import numpy as np

# 원본 NPZ 파일 경로
original_npz_path = '/home/hwang-gyuhan/Workspace/Transformer/preprocessed_filtered_data.npz'

# 새로 만들 NPZ 파일 경로
output_npz_path = '/home/hwang-gyuhan/Workspace/Transformer/preprocessed_data_with_labels.npz'

# 원본 NPZ 파일 로드
data = np.load(original_npz_path)

# track_id와 genre 추출
track_ids = data['track_id']
genre = data['genre']

# 원핫 인코딩을 정수 레이블로 변환
genre_labels = np.argmax(genre, axis=1)

# 제외할 track_id 목록
excluded_track_ids = ['98565', '98567', '98569', '99134', '108925', '133297']

# 제외할 track_id를 제외한 인덱스 찾기
valid_indices = [i for i, track_id in enumerate(track_ids.astype(str)) if track_id not in excluded_track_ids]

# 새로운 데이터로 필터링
filtered_track_ids = track_ids[valid_indices]
filtered_genre_labels = genre_labels[valid_indices]

# 새로운 NPZ 파일로 저장 (정수 레이블로 변환된 genre를 저장)
np.savez(output_npz_path, track_id=filtered_track_ids, genre=filtered_genre_labels)

# 새로 만든 NPZ 파일의 내용을 확인
new_data = np.load(output_npz_path)

# 새로운 NPZ 파일에 포함된 키들 출력
print("새로운 NPZ 파일에 포함된 키들:", new_data.files)

# 각 키에 해당하는 데이터 크기와 내용을 출력
for key in new_data.files:
    print(f"키: {key}, 데이터 크기: {new_data[key].shape}")
    print(f"데이터 내용 (첫 5개 항목):\n{new_data[key][:5]}")  # 첫 5개 항목만 출력