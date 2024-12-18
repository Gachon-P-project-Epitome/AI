from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np
import scipy
import json
from sklearn.utils import check_array
import pandas as pd
import warnings


# TorchDataset은 Dataset 클래스를 상속받아 PyTorch에서 사용할 데이터셋 클래스를 정의
# 이 클래스는 2D NumPy 배열을 입력받아 처리하는데 사용됨
# X는 입력 행렬, y는 one-hot 인코딩된 타겟 
class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    # 생성자 메서드로, x와 y를 클래스 속성으로 저장 
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # __len__ 메서드는 데이터셋의 길이(샘플 수)를 반환, x의 길이를 기준으로함
    def __len__(self):
        return len(self.x)


    # __getitem__ 메서드는 주어진 인덱스에 해당하는 입력 x 와 타겟 y를 반환
    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


# SparseTorchDataset은 Dataset 클래스를 상속받아 CSR(Compressed Sparse Row) 행렬을 처리하는 데이터셋 클래스 
# x는 CSR 행렬 형식의 입력 행렬, y는 one-hot 인코딩된 타겟 
class SparseTorchDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    # 생성자 메서드로, CSR 행렬 x와 y클래스 속성으로 저장함
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # __len__ 메서드는 CSR 행렬 x의 첫 번째 차원의 크기(행 수)를 반환함
    def __len__(self):
        return self.x.shape[0]

    # __getitme__ 메서드는 주어진 인덱스에 해당하는 CSR 행렬의 행을 NumPy 배열로 변환 후
    # PyTorch의 Tensor로 변환하여 반환함. y는 해당 인덱스의 타겟 값을 반환
    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        y = self.y[index]
        return x, y


# PredictDataset은 Dataset 클래스를 상속받아 예측을 위한 데이터셋 클래스를 정의
# X는 입력 행렬 
class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    # 생성자 메서드로 입력 x를 클래스 속성으로 저장
    def __init__(self, x):
        self.x = x

    # __len__ 메서드는 데이터셋의 길이(샘플 수)를 반환. x의 길이를 기준으로 함
    def __len__(self):
        return len(self.x)

    # __getitem__ 메서드는 주어진 인덱스에 해당하는 입력 x를 반환
    def __getitem__(self, index):
        x = self.x[index]
        return x


# SparsePreictDataset 클래스는 PyTorch의 Dataset 클래스를 상속받아 CSR(Compressed Sparse Row) 행렬을 입력으로 받는 데이터셋을 정의
# x는 CSR 형식의 입력 행렬 
class SparsePredictDataset(Dataset):
    """
    Format for csr_matrix

    Parameters
    ----------
    X : CSR matrix
        The input matrix
    """

    # 생성자 메서드로 CSR 행렬 x를 클래스의 속성으로 저장 
    def __init__(self, x):
        self.x = x

    # __len__ 메서드는 CSR 행렬 x의 첫 번째 차원의 크기(행 수)를 반환
    def __len__(self):
        return self.x.shape[0]

    # __getitem__메서드는 주어진 인덱스에 해당하는 CSR 행렬의 행을 NumPy 배열로 변환하고 PyTorch의 Tensor로 변환하여 반환
    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index].toarray()[0]).float()
        return x


# create_sampler 함수는 주어진 가중치에 따라 샘플러를 생성
# weights는 가중치 설정을 정의
# 0: 가중치 없음
# 1: 클래스 불균형을 보정하기 위해 클래스의 역 빈도수에 기반한 가중치 적용
# 딕셔너리: 클래스 값에 대해 샘플 가중치 설정
# 반복 가능한 객체: 훈련 샘플 수와 동일한 길이의 리스트 또는 배열 
def create_sampler(weights, y_train):
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    y_train : np.array
        Training targets
    """
    
    # weights가 정수일 경우의 처리
    # weights가 0이면, 가중치를 적용하지 않으며 데이터 샘플을 셔플할 필요가 있다고 설정 (need_shuffle = True)
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
            
        # weights가 1일 경우
        # 데이터가 셔플될 필요가 없으며 (need_shuffle = False), 클래스의 샘플 수를 계산 
        # 클래스 샘플 수의 역수로 가중치를 설정
        # 타겟 y_train의 각 클래스에 대해 가중치를 할당
        # 이 가중치를 PyTorch의 WeightedRansomSampler에 전달하여 샘플러 생성
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            
        # weights가 0이나 1이 아닌 경우, 유효하지 않은 값으로 오류를 발생시킴 
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
        
    # weight가 딕셔너리일 경우
    # 데이터가 셔플될 필요가 없으며 (need_shuffle = False)
    # 각 클래스에 대해 정의된 가중치를 y_train에 적용함
    # WeightedRansomSampler를 사용하여 샘플러를 생성 
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
    # weights가 반복 가능한 객체일 경우
    # weights의 길이가 y_train의 길이와 동일해야 함
    # need_shuffle을 Flase로 설정하고, 가중치 배열로 변환하여 WeightedRandomSampler를 사용하여 샘플러를 생성 
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
           
    return need_shuffle, sampler


# create_dataloaders 함수는 주어진 매개변수에 따라 훈련과 검증 데이터 로더를 생성
def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.array
        Mapped Training targets
    eval_set : list of tuple
        List of eval tuple set (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    
    # create_sampler 함수를 호출하여 가중치와 샘플러를 생성
    # need_shuffle은 데이터 셔플 필요 여부를 나타내고, sampler는 샘플링 방법을 나타냄 
    need_shuffle, sampler = create_sampler(weights, y_train)

    # X_train이 CSR행렬인 경우
    # SparseTorchDatasets을 사용하여 훈련 데이터셋을 생성
    # DataLoader를 통해 배치크기, 샘플러, 셔플 여부, 서브 프로세스 수, 마지막 배치 처리 방식, GPU 메모리 고정 여부를 설정 
    if scipy.sparse.issparse(X_train):
        train_dataloader = DataLoader(
            SparseTorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        
    # X_train이 CSR 행렬이 아닌 경우
    # TorchDataset을 사용하여 훈련 데이터셋을 생성함
    # Dataloader의 매개변수는 위와 같다 
    else:
        train_dataloader = DataLoader(
            TorchDataset(X_train.astype(np.float32), y_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    # eval_set의 각 (X, y)쌍에 대해 
    # X가 CSR 행렬인 경우
    # SparseTorchDataset을 사용하여 검증 데이터셋을 생성
    # DataLoader를 통해 배치 크기, 셔플 여부, 서브 프로세스 수, GPU 메로리 고정 여부를 설정 
    valid_dataloaders = []
    for X, y in eval_set:
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparseTorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )
            
        # X가 CSR 행렬이 아닌경우
        # TorchDataset을 사용하여 검증 데이터셋을 생성
        # DataLoader의 매개변수는 위와 같다 
        else:
            valid_dataloaders.append(
                DataLoader(
                    TorchDataset(X.astype(np.float32), y),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
            )

    # 생성된 훈련 데이터 로더와 검증 데이터 로더 목록을 반환 
    return train_dataloader, valid_dataloaders


# create_explain_matrix 함수는 동일한 임베딩에서 중요성을 빠르게 합산하기 위한 계산적인 트릭을 사용하여 매핑 행렬을 생성
# input_dim은 초기 입력 차원, cat_emb_dim은 범주형 특성의 임베딩 크기
# cat_idxs는 범주형 특성의 초기 위치, post_embed_dim은 임베딩 후 입력 차원
def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce
    """

    # cat_emb_dim이 정수일 경우
    # 모든 범주형 특성에 대해 같은 크기의 임베딩을 사용하며, 각 범주형 특성의 영향도를 계산
    
    # cat_emb_dim이 리스트일 경우
    # 각 범주형 특성의 임베딩 크기에서 1을 뺀 값을 사용하여 영향도를 계산 
    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    # input_dim의 각 인덱스에 대해
    # cat_idxs에 포함되지 않은 경우: 단일 인덱스를 indices_trick에 추가
    # cat_idxs에 포함된 경우: 해당 범주형 특성의 임베딩 범위를 Indices_trick에 추가하고, acc_emb와 nb_emb를 업데이트 
    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    # post_embed_dim X input_dim 크기의 제로 행렬을 생성
    # indices_trick의 각 항목에 대해, 해당하는 열을 1로 설정하여 매핑 행렬을 생성
    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    # 생성된 행렬을 CSC(Compressed Sparse Column) 형식의 희소 행렬로 변환하여 반환
    return scipy.sparse.csc_matrix(reducing_matrix)


# create_group_matrix 함수는 주어진 list_groups에 따라 그룹 행렬을 생성
# list_groups는 각 그룹에 속한 특성들의 리스트를 포함하는 리스트
# input_dim은 초기 데이터셋의 특성 수
# 반환값은 group_matrix로, 각 그룹의 특성 중요도를 나타내는 행렬
def create_group_matrix(list_groups, input_dim):
    """
    Create the group matrix corresponding to the given list_groups

    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset

    Returns
    -------
    - group_matrix : torch matrix
        A matrix of size (n_groups, input_dim)
        where m_ij represents the importance of feature j in group i
        The rows must some to 1 as each group is equally important a priori.

    """
    # check_list_groups 함수를 호출하여 List_groups의 유효성을 검사 
    check_list_groups(list_groups, input_dim)

    # list_groups가 비어있는 경우
    # 단위 행렬(torch.eye)을 생성하여 반환
    # 단위 행렬은 모든 특성이 독립적인 그룹을 형성할 때 사용됨
    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    
    # list_groups에 그룹이 있는 경우
    # n_groups는 그룹의 수를 계산, 각 그룹의 특성 수에서 1을 빼고 전체 특성 수에서 이를 뺀 값을 사용
    # group_matrix를 n_groups X input_dim 크기의 제로 행렬로 초기화 함
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        # 아직 그룹에 포함되지 않은 특성의 인덱스를 저장 
        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        # list_groups의 각 그룹에 대해
        # current_group_idx는 현재 그룹의 인덱스를 추적
        # 각 그룹의 특성에 대해 중요도를 1/그룹 크기로 설정하고, 해당 특성을 remaining_features에서 제거
        # 모든 특성에 대해 반복한 후, current_group_idx를 증가시켜 다음 그룹으로 이동 
        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
            
            
        # features not mentionned in list_groups get assigned their own group of singleton
        # list_groups에 포함되지 않은 특성에 대해
        # 각 특성에 대해 중요도를 1로 설정하고, 새 그룹을 만들어 이를 group_matrix에 추가
        # 모든 남은 특성에 대해 반복한 후, current_group_idx를 증가 시킴 
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
            
        # 생성된 group_matrix를 반환 
        return group_matrix


# check_list_groups 함수는 list_groups의 유효성을 검사
# list_groups는 특성 그룹을 나타내는 리스트의 리스트
# input_dim은 초기 데이터셋의 특성 수
def check_list_groups(list_groups, input_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= input_dim)
        - does not contain empty groups
    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assign a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset
    """
    
    # list_groups가 리스트인지 확인 
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    # list_groups가 비어 있으면 함수 종료 
    if len(list_groups) == 0:
        return
    
    # list_groups의 각 그룹에 대해
    # 그룹이 리스트인지 확인
    # 그룹이 비어 있지 않은지 확인 
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    # list_groups의 모든 요소를 평탄화 하여:
    # n_elements_in_groups는 list_groups에 있는 모든 특성의 수
    # flat_list는 list_groups의 모든 특성 인덱스를 포함함
    # unique_elements는 중복을 제거한 특성 인덱스 
    # n_unique_elements_in_groups는 유니크한 특성 수
    # 유니크 특성 수가 그룹에 있는 특성 수와 일치하는지 확인 
    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = f"One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    # unique_elements에서 최대 특성 인덱스를 계산하여:
    # 이 값이 input_dim보다 작아야 함
    # 특성 인덱스가 데이터셋의 특성 수를 초과하지 않도록 함
    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return


# filter_weights 함수는 weights 매개변수가 올바른 형식인지 확인
# weights는 정수, 딕셔너리, 또는 리스트일 수 있음
# 함수는 형식이 잘못된 경우에만 오류를 발생시키고, 올바르면 아무것도 반환하지 않음
def filter_weights(weights):
    """
    This function makes sure that weights are in correct format for
    regression and multitask TabNet

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Returns
    -------
    None : This function will only throw an error if format is wrong
    """
    
    # err_msg는 오류 메시지의 시작 부분을 설정 
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    
    # weights가 정수형인 경우:
    # 값이 1이면, 해당 오류 메시지와 함께 ValueError를 발생시킴
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
        
    # weights가 딕셔너리형인 경우:
    # 해당 오류 메시지와 함께 ValueError를 발생시킴 
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    
    # 함수가 아무것도 반환하지 않고 종료됨
    return


# validate_eval_set 함수는 eval_set의 형태가 (X_train, y_train)과 호환되는지 확인
# eval_set은 튜플(X, y)의 리스트
# eval_name은 평가 세트의 이름을 포함하는 리스트
# X_train과 y_train은 각각 훈련데이터와 라벨
# 반환값은 검증된 eval_names와 eval_set
def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    
    # eval_name이 비어있으면, eval_set의 길이에 따라 기본 이름을 생성 
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    # eval_set과 eval_name의 길이가 같아야 함을 확인 
    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    
    # eval_set의 모든 튜플이 두 개의 요소를 가져야 함을 확인
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
        
    # eval_name과 eval_set의 각 쌍에 대해 반복하며
    # X에 대해 check_input 함수를 호출하여 입력 데이터의 유효성을 검사 
    for name, (X, y) in zip(eval_name, eval_set):
        check_input(X)
        
        # X와 X_train의 차원수가 일치하는지 확인
        # 일치하지 않으면 오류 메시지를 출력
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        # y와 y_train의 차원 수가 일치하는지 확인
        # 일치하지 않으면 오류 메시지를 출력 
        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        # X와 X_train의 열 수가 일치하는지 확인
        # 일치하지 않으면 오류 메시지를 출력 
        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        # y_train이 2차원일 경우:
        # y와 y_train의 열 수가 일치하는지 확인
        # 일치하지 않으면 오류 메시지를 출력 
        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
            
        # X와 y의 행 수가 일치하는지 확인
        # 일치하지 않으면 오류 메시지를 출력 
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    # 검증된 eval_name과 eval_set을 반환 
    return eval_name, eval_set

# define_device 함수는 훈련 및 추론에 사용할 장치를 정의
# device_name은 "auto", "cpu", "cuda"중 하나일 수 있다 
# 반환값은 "cpu", "cuda" 
def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    
    # device_name이 "auto"인 경우:
    # CUDA가 사용 가능한지 확인하고, 사용 가능하면 "cuda"를 반환
    # 사용 불가능하면 "cpu"를 반환
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
        
    # device_name이 "cuda"이고 CUDA가 사용 불가능한 경우:
    # "cpu"를 반환 
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    
    # device_namd이 "auto"또는 "cuda"가 아닌 경우:
    # 주어진 device_name을 그대로 반환 
    else:
        return device_name


# ComplexEncoder 클래스는 json.JSONEncoder를 상속받아 JSON 인코딩을 커스터마이즈함
class ComplexEncoder(json.JSONEncoder):
    
    # default 메서드는 json.JSONEncoder의 기본 메서드를 오버라이드하여, 객체를 JSON으로 직렬화할 때 사용
    def default(self, obj):
        
        # obj가 NumPy의 generic 타입이나 ndarray인 경우:
        # tolist() 메서드를 호출하여 Python 리스트로 변환
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        
        # Let the base class default method raise the TypeError
        # obj가 위의 조건에 해당하지 않으면, 기본 JSONEncoder의 default 메서드를 호출하여 TypeError를 발생시킴
        return json.JSONEncoder.default(self, obj)


# check_input 함수는 X가 Pandas DataFrame일 경우 명확한 오류를 발생시키고
# X가 Scikit-learn 규칙에 맞는지 확인함
def check_input(X):
    """
    Raise a clear error if X is a pandas dataframe
    and check array according to scikit rules
    """
    
    # X가 Pandas DataFrame이나 Series인 경우 오류 메시지를 발생시킴 
    if isinstance(X, (pd.DataFrame, pd.Series)):
        err_message = "Pandas DataFrame are not supported: apply X.values when calling fit"
        raise TypeError(err_message)
    
    # check_array 함수를 호출하여 X가 Scikit-learn 배열 규칙에 맞는지 검사
    # accept_sparse=True로 설정하여 희소 행렬도 허용
    check_array(X, accept_sparse=True)


# check_warm_start 함수는 두 매개변수의 애매한 사용에 대해 경고 
def check_warm_start(warm_start, from_unsupervised):
    """
    Gives a warning about ambiguous usage of the two parameters.
    """
    
    # warm_start가 True이고 from_unsupervised가 None이 아닌 경우 에러메시지 발생시킴
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg = "warm_start will be ignore, training will start from unsupervised weights"
        warnings.warn(warn_msg)
        
    # 함수가 아무것도 반환하지 않고 종료됨
    return


# check_embedding_parameters 함수는 임베딩 관련 파라미터를 확인하고, 고유한 방식으로 재정렬
def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    """
    Check parameters related to embeddings and rearrange them in a unique manner.
    """
    
    # cat_dims와 cat_idxs중 하나만 비어 있는 경우:
    # 둘다 비어있지 않거나 둘다 비어있어야 한다는 오류 메시지를 설정하고, ValueError를 발생시킴 
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    
    # cat_dims와 cat_idxs의 길이가 다르면:
    # cat_dims와 cat_idxs는 같은 길이를 가져야 한다는 오류 메시지를 설정하고 ValueError를 발생시킴 
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    # cat_emv_dim이 정수인 경우:
    # 모든 범주형 특성에 대해 동일한 크기의 임베딩을 설정함
    # 그렇지 않으면 cat_emb_dim을 그대로 사용하여 임베딩 크기를 설정
    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim


    # check that all embeddings are provided
    # cat_emb_dims와 cat_dims의 길이가 다르면
    # cat_emb_dim과 cat_dims는 같은 기이의 리스트여야 한다라는 오류 메시지를 설정하고 ValueError를 발생시킴 
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)


    # Rearrange to get reproducible seeds with different ordering
    # cat_idxs가 비어있지 않으면:
    # cat_idxs를 정렬하여 cat_dims와 cat_emb_dims를 정렬된 인덱스에 맞게 재배열함
    # 이렇게 하면 시드가 재현 가능하도록 함
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    # 재배열된 cat_dims, cat_idxs, cat_emb_dims를 반환 
    return cat_dims, cat_idxs, cat_emb_dims
