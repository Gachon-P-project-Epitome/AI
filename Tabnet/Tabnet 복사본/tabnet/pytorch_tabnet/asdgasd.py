import sys
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# tab_network.py 파일이 있는 경로를 추가
sys.path.append('/home/hwang-gyuhan/Workspace/Tabnet/Tabnet 복사본/tabnet/pytorch_tabnet')
from tab_network import EmbeddingGenerator  # EmbeddingGenerator 클래스 임포트

# 가상의 레이블 데이터
labels = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 임베딩 생성에 필요한 매개변수
input_dim = len(encoded_labels)  # 레이블 수
cat_dims = [len(label_encoder.classes_)]  # 레이블 클래스 수
cat_emb_dims = [4]  # 임베딩 차원
cat_idxs = [0]  # 레이블 인덱스
group_matrix = torch.zeros((1, 1))  # 그룹 행렬 초기화 (예를 들어)

# EmbeddingGenerator 인스턴스 생성
embedding_generator = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix)

# 레이블을 임베딩
label_tensor = torch.tensor(encoded_labels).unsqueeze(1).float()
print("Label Tensor Shape:", label_tensor.shape)  # 텐서의 크기 확인
embedded_labels = embedding_generator(label_tensor)
print(embedded_labels)