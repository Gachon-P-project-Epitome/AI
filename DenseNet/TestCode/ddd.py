import numpy as np

# NPZ 파일 경로
npz_file_path = "/home/hwang-gyuhan/Workspace/dataset/vector/Instrumental.npz"

# NPZ 파일 로드
data = np.load(npz_file_path, allow_pickle=True)

# 필요한 키
features_key = 'features'  # 벡터 데이터의 키
file_names_key = 'file_names'  # 파일 이름 데이터의 키

# 확인하려는 인덱스
target_index = 793

# 키가 있는지 확인
if features_key in data and file_names_key in data:
    features = data[features_key]
    file_names = data[file_names_key]
    
    # 인덱스 범위 확인
    if 0 <= target_index < len(file_names):
        print(f"File name at index {target_index}: {file_names[target_index]}")
    else:
        print(f"Index {target_index} is out of bounds. Total files: {len(file_names)}")
else:
    print(f"'{features_key}' 또는 '{file_names_key}' 키가 NPZ 파일에 없습니다.")