import numpy as np

# NPZ 파일 경로
npz_file_path = '/home/hwang-gyuhan/Workspace/dataset/vector/Rock.npz'

# NPZ 파일 로드
data = np.load(npz_file_path, allow_pickle=True)

# 'features'와 'file_names' 데이터 확인
features = data['features']
file_names = data['file_names']

# 10개 샘플 추출
num_samples = 10
if features.shape[0] < num_samples:
    num_samples = features.shape[0]  # 데이터가 10개 미만인 경우 조정

# 10개 샘플 선택

selected_file_names = file_names[:num_samples]


print("\nSelected File Names:")
print(selected_file_names)
