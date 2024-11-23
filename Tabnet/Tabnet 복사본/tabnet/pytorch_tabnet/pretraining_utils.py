from torch.utils.data import DataLoader
from pytorch_tabnet.utils import (
    create_sampler,
    SparsePredictDataset,
    PredictDataset,
    check_input
)
import scipy


# create_dataloaders 함수는 가중치에 따라 서브샘플링이 적용된 데이터로더를 생성 
def create_dataloaders(
    X_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray or scipy.sparse.csr_matrix
        Training data
    eval_set : list of np.array (for Xs and ys) or scipy.sparse.csr_matrix (for Xs)
        List of eval sets
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
    
    # create_sampler 함수를 호출하여 데이터 샘플링 전략과 데이터 셔플 여부를 결정
    # weights와 X_train을 인자로 전달 
    need_shuffle, sampler = create_sampler(weights, X_train)

    # X_train이 희소 행렬인지 확인
    if scipy.sparse.issparse(X_train):
        
        # X_train이 희소 행렬인 경우:
        # SparsePredictDataset을 사용하여 train_dataloader를 생성 
        train_dataloader = DataLoader(
            SparsePredictDataset(X_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        
    # X_train이 희소 행렬이 아닌 경우:
    # PredictDataset을 사용하여 train_dataloader를 생성 
    else:
        train_dataloader = DataLoader(
            PredictDataset(X_train),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=need_shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    # eval_set의 각 평가 데이터를 반복하여 valid_dataloaders 리스트를 생성 
    valid_dataloaders = []
    for X in eval_set:
        
        # X가 희소 행렬인 경우:
        # SparsePredictDataset을 사용하여 valid_dataloaders 리스트에 데이터로더를 추가
        if scipy.sparse.issparse(X):
            valid_dataloaders.append(
                DataLoader(
                    SparsePredictDataset(X),
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=need_shuffle,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    pin_memory=pin_memory,
                )
            )
            
        # X가 희소 행렬이 아닌 경우:
        # PredictDataset을 사용하여 valid_dataloaders 리스트에 데이터로더를 추가
        else:
            valid_dataloaders.append(
                DataLoader(
                    PredictDataset(X),
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=need_shuffle,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    pin_memory=pin_memory,
                )
            )

    # train_dataloader와 valid_dataloaders를 반환
    return train_dataloader, valid_dataloaders


# validate_eval_set 함수는 eval_set의 형태가 X_train과 호환되는지 확인
def validate_eval_set(eval_set, eval_name, X_train):
    """Check if the shapes of eval_set are compatible with X_train.

    Parameters
    ----------
    eval_set : List of numpy array
        The list evaluation set.
        The last one is used for early stopping
    X_train : np.ndarray
        Train owned products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.

    """
    
    # eval_name이 제공되지 않은 경우, 기본적으로 "val_0","val_1"등의 이름을 생성
    eval_names = eval_name or [f"val_{i}" for i in range(len(eval_set))]
    
    # eval_set과 eval_names의 길이가 같은지 확인, 그렇지 않으면 오류를 발생시킴 
    assert len(eval_set) == len(
        eval_names
    ), "eval_set and eval_name have not the same length"

    # eval_set의 각 평가 데이터를 반복, set_nb는 현재 평가 데이터의 인덱스 
    for set_nb, X in enumerate(eval_set):
        
        # check_input 함수를 호출하여 X가 올바른 형식인지 확인 
        check_input(X)
        
        # 평가 데이터 X의 열 수가 X_train의 열 수와 다른지 확인
        # 열 수가 다르면 오류 메시지를 설정, AssertionError를 발생시킴 
        msg = (
            f"Number of columns is different between eval set {set_nb}"
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg
        
    # 검증된 eval_names를 반환 
    return eval_names
