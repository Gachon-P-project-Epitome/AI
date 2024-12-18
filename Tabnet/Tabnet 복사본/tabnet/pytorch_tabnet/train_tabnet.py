from dataclasses import dataclass, field
from typing import List, Any, Dict
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from abc import abstractmethod
import matplotlib.pyplot as plt

from pytorch_tabnet import tab_network
from pytorch_tabnet.utils import (
    SparsePredictDataset,
    PredictDataset,
    create_explain_matrix,
    validate_eval_set,
    create_dataloaders,
    define_device,
    ComplexEncoder,
    check_input,
    check_warm_start,
    create_group_matrix,
    check_embedding_parameters
)
from pytorch_tabnet.callbacks import (
    CallbackContainer,
    History,
    EarlyStopping,
    LRSchedulerCallback,
)
from pytorch_tabnet.metrics import MetricContainer, check_metrics
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pytorch_tabnet.tab_network import (
    TabNetEncoder,
    TabNetDecoder,
    initialize_glu,
    GBN,
    FeatTransformer,
    GLU_Block,
    GLU_Layer,
    AttentiveTransformer,
)

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

from torch.utils.data import DataLoader
import io
import json
from pathlib import Path
import shutil
import zipfile
import warnings
import copy
import scipy
import os









def load_data(features_csv_path: str, targets_npz_path: str):
    # CSV 파일로부터 피처 데이터 로드
    features = pd.read_csv(features_csv_path, header=3, index_col=0)  # 4행부터 데이터 시작
    # NPZ 파일로부터 타겟 데이터 로드
    targets = np.load(targets_npz_path)
    y = targets['genre']  # 'genre' 키로 타겟 배열 추출

    # NaN 값을 가진 track_id 식별
    nan_track_ids = features[features.isnull().any(axis=1)].index.tolist()

    # NaN track_id가 있는 행 제외
    features_cleaned = features[~features.index.isin(nan_track_ids)].values
    y_cleaned = y[~np.isin(targets['track_id'], nan_track_ids)]  # y에서 동일한 track_id 제외

    return features_cleaned, y_cleaned

def preprocess_target(y):
    # One-Hot Encoding된 배열을 그대로 반환
    return y  # y는 이미 One-Hot Encoding 되어 있음

def split_data(features, targets):
    # 데이터 나누기 (8:2 비율)
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, stratify=targets)

    # 타겟 데이터 전처리
    y_train = preprocess_target(y_train)
    y_test = preprocess_target(y_test)

    # 각 세트의 shape 출력
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return {
        'train': X_train,
        'test': X_test
    }, {
        'train': y_train,
        'test': y_test
    }

# 피처 및 타겟 데이터 경로
features_csv_path = '/home/hwang-gyuhan/Workspace/Tabnet/Tabnet 복사본/features.csv'  # CSV 파일 경로
targets_npz_path = '/home/hwang-gyuhan/Workspace/Tabnet/track_genre_data.npz'    # NPZ 파일 경로

# 데이터 로드
features, targets = load_data(features_csv_path, targets_npz_path)

# 데이터 분할
X_dict, y_dict = split_data(features, targets)

# CSR 행렬로 변환
sparse_X_train = scipy.sparse.csr_matrix(X_dict['train'])  # Create a CSR matrix from X_train
sparse_X_test = scipy.sparse.csr_matrix(X_dict['test'])    # Create a CSR matrix from X_test

# 레이블 변환
y_train = np.argmax(y_dict['train'], axis=1)  # One-Hot Encoding된 y를 인덱스로 변환
y_test = np.argmax(y_dict['test'], axis=1)

tabnet_params = {
    "cat_idxs": [],
    "cat_dims": [],
    "cat_emb_dim": 1,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=1e-4),
    "scheduler_fn": None,
    "mask_type": 'sparsemax',
    "device_name": 'cuda',
    "n_d": 8,
    "n_a": 8,
    "n_steps": 2,
    "gamma": 1.3,
    "seed": 21
}

clf = TabNetClassifier(**tabnet_params)

max_epochs = 1000

# 모델 훈련
clf.fit(
    X_train=sparse_X_train,
    y_train=y_train,  # One-Hot Encoding된 타겟 사용
    eval_set=[(sparse_X_train, y_train), (sparse_X_test, y_test)],
    eval_name=['train', 'val'],
    eval_metric=['accuracy', 'logloss'],
    max_epochs=max_epochs,
    patience=0,
    batch_size=1024,
    virtual_batch_size=128,
)

# 손실 및 정확도 그래프 시각화
plt.figure(figsize=(14, 6))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(clf.history['loss'], marker='o', label='train loss')
plt.plot(clf.history['val_logloss'], marker='o', label='val loss')
plt.title('Loss per epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(clf.history['train_accuracy'], marker='o', label='train accuracy')
plt.plot(clf.history['val_accuracy'], marker='o', label='val accuracy')
plt.title('Accuracy per epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()

plt.savefig('training_graph.png')  # 'training_graph.png' 파일로 저장
plt.close()