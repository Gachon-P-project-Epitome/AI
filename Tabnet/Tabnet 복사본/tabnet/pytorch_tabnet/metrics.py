from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error,
)
import torch


# 비지도 손실 함수를 정의 y_pred는 예측된 값, embedded_x는 임베딩된 원래 입력
# obf_vars는 어느 변수가 오버스크루드(가려진)되었는지 나타내는 바이너리 마스크
# eps는 작은 값으로, 나눗셈 시 0으로 나누는 것을 방지 
def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value

    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    
    # 예측 된 값과 원래 입력 간의 차이를 계산하여 errors에 저장, 이것이 재구성 오류
    errors = y_pred - embedded_x
    
    # obs_vars 마스크를 사용하여 가려진 변수를 기준으로 재구성 오류를 선택하고, 제곱하여 reconstruction_errors에 저장 
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    
    # 입력된 데이터의 각 피처에 대한 평균값을 계산하여 batch_means에 저장 
    batch_means = torch.mean(embedded_x, dim=0)
    
    # 평균값이 0인 경우 1로 변경하여 나눗셈에서 0으로 나누는 것을 방지
    batch_means[batch_means == 0] = 1

    # 각 피처에 대한 표준 편차를 계산하고, 이를 제곱하여 batch_stds에 저장 (분산을 구하는 것과 동일)
    batch_stds = torch.std(embedded_x, dim=0) ** 2
    
    # 표준 편차가 0인 경우 분산이 없는 피처는 평균값으로 대체 
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    
    # 재구성 오류를 표준 편차로 나누어 정규화한 후, 피처 손실을 계산
    features_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    
    # compute the number of obfuscated variables to reconstruct
    # 오버스크루드된(가려진)변수를 재구성해야 하는 수를 계산하여 nb_reconstructed_variables에 저장 
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    
    # take the mean of the reconstructed variable errors
    # 재구성 오류를 재구성해야 할 변수의 수로 나누어 평균값을 계산, 여기서 eps는 0으로 나누는 것을 방지하기 위한 작은 값
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    
    # here we take the mean per batch, contrary to the paper
    # 전체 배치에서 평균 손실 값을 계산 
    loss = torch.mean(features_loss)
    
    # 최종 손실값을 반환 
    return loss


# 비지도 손실 함수를 정의
# y_pred는 예측된 값, embedded_x는 임베딩된 원래 입력, obf_vars는 어느 변수가 오버스크루드(가려진)되었는지 나타내는 바이너리 마스크
# eps는 작은 값으로, 나눗셈 시 0으로 나누는 것을 방지
def UnsupervisedLossNumpy(y_pred, embedded_x, obf_vars, eps=1e-9):
    
    # 예측된 값과 원래 입력 간의 차이를 계산하여 errors에 저장, 이것이 재구성 오류
    errors = y_pred - embedded_x
    
    # obf_vars 마스크를 사용하여 가려진 변수를 기준으로 재구성 오류를 선택하고, 제곱하여 reconstruction_errors에 저장 
    reconstruction_errors = np.multiply(errors, obf_vars) ** 2
    
    # 입력된 데이터의 각 피처에 대한 평균값을 계산하여 batch_means에 저장 
    batch_means = np.mean(embedded_x, axis=0)
    
    # 평균값이 0인 경우 1로 변경하여 나눗셈에서 0으로 나누는 것을 방지 
    batch_means = np.where(batch_means == 0, 1, batch_means)

    # 각 피처에 대한 표준 편차를 계산하고, 이를 제곱하여 batch_stds에 저장
    batch_stds = np.std(embedded_x, axis=0, ddof=1) ** 2
    
    # 표준편차가 0인 경우 분산이 없는 피처는 평균값으로 대체 
    batch_stds = np.where(batch_stds == 0, batch_means, batch_stds)
    
    # 재구성 오류를 표준 편차로 나누어 정규화한 후 , 피처 손실을 계산
    features_loss = np.matmul(reconstruction_errors, 1 / batch_stds)
    
    # compute the number of obfuscated variables to reconstruct
    # 오버스크루드된(가려진)변수를 재구성해야 하는 수를 계산하여 nb_reconstructed_variables에 저장 
    nb_reconstructed_variables = np.sum(obf_vars, axis=1)
    
    # take the mean of the reconstructed variable errors
    # 재구성 오류를 재구성해야 할 변수의 수로 나누어 평균값을 계산, 여기서 eps는 0으로 나누는 것을 방지하기 위한 작은 값
    features_loss = features_loss / (nb_reconstructed_variables + eps)
    
    # here we take the mean per batch, contrary to the paper
    # 전체 배치에서 평균 손실 값을 계산 
    loss = np.mean(features_loss)
    
    # 최종적으로 계산된 비지도 손실 값을 반환
    return loss


# 비지도 학습에서 메트릭을 저장하고 관리하기 위한 컨테이너 클래스를 정의 
@dataclass
class UnsupMetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variables was obfuscated so reconstruction is based on this.

    """

    # metric_names 는 사용할 메트릭 이름들의 리스트
    # prefix는 메트릭 이름 앞에 붙일 문자열로, 기본값은 빈 문자열 
    metric_names: List[str]
    prefix: str = ""

    # 클래스 초기화 후에 호출되는 메서드 
    def __post_init__(self):
        
        # Metric.get_metrics_by_names() 메서드를 통해 메트릭 이름에 해당하는 메트릭 객체들을 가져와 저장 
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        
        # 메트릭 이름에 prefix를 붙여 self.names에 저장
        self.names = [self.prefix + name for name in self.metric_names]

    
    # 클래스 인스턴스를 함수처럼 호출됐을 때 실행되는 메서드
    # 비지도 학습에 필요한 예측값, 임베딩된 입력, 오버스크루드된 변수를 받아 메트릭을 계산 
    def __call__(self, y_pred, embedded_x, obf_vars):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        
        # 메트릭 이름과 그 값을 저장할 딕셔너리를 초기화 
        logs = {}
        
        # 메트릭 리스트를 순회하면서 각 메트릭을 계산 
        for metric in self.metrics:
            
            # 각 메트릭에 대해 예측값, 임베딩된 입력, 오버스크루드된 변수를 사용해 값을 계산 
            res = metric(y_pred, embedded_x, obf_vars)
            
            # 계산된 메트릭 값을 logs 딕셔너리에 저장 
            logs[self.prefix + metric._name] = res
            
        # 모든 메트릭이 포함된 딕셔너리를 반환 
        return logs


# 지도 학습에서 메트릭을 저장하고 관리하기 위한 컨테이너 클래스를 정의 
@dataclass
class MetricContainer:
    """Container holding a list of metrics.

    Parameters
    ----------
    metric_names : list of str
        List of metric names.
    prefix : str
        Prefix of metric names.

    """

    # metric_names는 사용할 메트릭 이름들의 리스트
    # prefix는 메트릭 이름 앞에 붙일 문자열로, 기본값은 빈 문자열 
    metric_names: List[str]
    prefix: str = ""

    # 클래스 초기화 후에 호출되는 메서드 
    def __post_init__(self):
        
        # Metric.get_metrics_by_names() 메서드를 통해 메트릭 이름에 해당하는 메트릭 객체들을 가져와 저장 
        self.metrics = Metric.get_metrics_by_names(self.metric_names)
        
        # 메트릭 이름에 prefix를 붙여 self.names 에 저장 
        self.names = [self.prefix + name for name in self.metric_names]

    # 클래스 인스턴스를 함수처럼 호출했을 때 실행되는 메서드
    # 실제 값 y_true와 예측된 값 y_pred를 받아 메트릭을 계산 
    def __call__(self, y_true, y_pred):
        """Compute all metrics and store into a dict.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_pred : np.ndarray
            Score matrix or vector

        Returns
        -------
        dict
            Dict of metrics ({metric_name: metric_value}).

        """
        
        # 메트릭 이름과 그 값을 저장할 딕셔너리를 초기화 
        logs = {}
        
        # 메트릭 리스트를 순회하면서 각 메트릭을 계산 
        for metric in self.metrics:
            
            # y_pred가 리스트인 경우, 여러 예측값을 처리 
            if isinstance(y_pred, list):
                
                # 각 예측 값에 대해 메트릭을 계산하고, 그 평균값을 구함
                res = np.mean(
                    [metric(y_true[:, i], y_pred[i]) for i in range(len(y_pred))]
                )
                
            # y_pred가 리스트가 아닌 경우, 일반적인 방식으로 메트릭을 계산 
            else:
                res = metric(y_true, y_pred)
                
            # 계산된 메트릭 값을 logs 딕셔너리에 저장 
            logs[self.prefix + metric._name] = res
            
        # 모든 메트릭이 포함된 딕셔너리를 반환 
        return logs


# metric 클래스는 메트릭의 기본 틀을 제공하며, __call__ 메서드는 모든 서브 클래스가 구현해야할 메서드
# 이 메서드를 호출 하면 "이 메서드는 구현되어야 한다"는 에러를 발생시킴 
class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Custom Metrics must implement this function")

    # 주어진 메트릭 이름 리스트를 기반으로 해당 메트릭 클래스를 반환하는 클래스 메서드 
    @classmethod
    def get_metrics_by_names(cls, names):
        """Get list of metric classes.

        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.

        Returns
        -------
        metrics : list
            List of metric classes.

        """
        
        # 현재 클래스의 서브클래스(즉, metric을 상속한 모든 클래스)를 리스트로 가져옴
        available_metrics = cls.__subclasses__()
        
        # 각 서브클래스의 인스턴스를 만들고, 그 인스턴스의 _name 속성을 가져와 리스트로 만듬 
        available_names = [metric()._name for metric in available_metrics]
        
        # 빈 리스트를 생성하고, 주어진 names 리스트를 반복문으로 순회
        metrics = []
        for name in names:
            
            # name이 available_names에 없으면 에러를 발생시키고, 어떤 이름들이 사용 가능한지 메시지로 출력
            assert (
                name in available_names
            ), f"{name} is not available, choose in {available_names}"
            
            # name에 해당하는 인덱스를 찾아, 그 인덱스에 있는 메트릭 클래스를 인스턴스로 만들어 metrics 리스트에 추가 
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
            
        # 완성된 메트릭 인스턴스 리스트를 반환 
        return metrics


# AUC(Area Under the Curve)를 계산하는 메트릭 클래스를 정의
# _name 속성은 "auc"로 설정하고, 최적화를 위해 True값을 설정 (AUC는 높을수록 좋기 때문에 True)
class AUC(Metric):
    """
    AUC.
    """
    def __init__(self):
        self._name = "auc"
        self._maximize = True

    # __call__ 메서드는 AUC 점수를 계산, y_score에서 각 샘플의 두 번째 클래스 확률 값을 사용
    def __call__(self, y_true, y_score):
        """
        Compute AUC of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            AUC of predictions vs targets.
        """
        return roc_auc_score(y_true, y_score[:, 1])


# 정확도를 계산하는 메트릭 클래스 정의
# _name은 "accuracy", 최적화는 True로 설정
class Accuracy(Metric):
    """
    Accuracy.
    """
    def __init__(self):
        self._name = "accuracy"
        self._maximize = True
        
    # 가장 높은 확률을 가지는 클래스를 예측값으로 설정
    # accuracy_score를 사용해 정확도를 계산
    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return accuracy_score(y_true, y_pred)


# 균형 잡힌 정확도를 계산하는 메트릭 클래스를 정의
# _name은 "balanced_accuracy", 최적화는 True로 설정
class BalancedAccuracy(Metric):
    """
    Balanced Accuracy.
    """
    def __init__(self):
        self._name = "balanced_accuracy"
        self._maximize = True

    # balanced_accuracy_score를 사용해 균형 잡힌 정확도를 계산 
    def __call__(self, y_true, y_score):
        """
        Compute Accuracy of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            Accuracy of predictions vs targets.
        """
        y_pred = np.argmax(y_score, axis=1)
        return balanced_accuracy_score(y_true, y_pred)


# 로그 손실 (Log Loss)을 계산하는 메트릭 클래스를 정의
# _name은 "logloss", 최적화는 False로 설정 
class LogLoss(Metric):
    """
    LogLoss.
    """
    def __init__(self):
        self._name = "logloss"
        self._maximize = False

    # log_loss 함수를 사용해 로그 손실 값을 계산 
    def __call__(self, y_true, y_score):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            LogLoss of predictions vs targets.
        """
        return log_loss(y_true, y_score)


# 평균 절대 오차(MAE)를 계산하는 메트릭 클래스를 정의
# _name은 "mae", 최적화는 False로 설정 
class MAE(Metric):
    """
    Mean Absolute Error.
    """
    def __init__(self):
        self._name = "mae"
        self._maximize = False

    # mean_absolute_error 함수를 사용해 MAE 값을 계산
    def __call__(self, y_true, y_score):
        """
        Compute MAE (Mean Absolute Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MAE of predictions vs targets.
        """
        return mean_absolute_error(y_true, y_score)


# 평균 제곱 오차(MAE)를 계산하는 메트릭 클래스를 정의
# _name은 "mse", 최적화는 False로 설정 
class MSE(Metric):
    """
    Mean Squared Error.
    """
    def __init__(self):
        self._name = "mse"
        self._maximize = False

    # mean_squared_error 함수를 사용해 MSE 값을 계산 
    def __call__(self, y_true, y_score):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            MSE of predictions vs targets.
        """
        return mean_squared_error(y_true, y_score)


# 로그 제곱 오차(RMSLE)를 계산하는 메트릭 클래스를 정의
# _name은 "rmsle", 최적화는 False로 설정
class RMSLE(Metric):
    """
    Root Mean squared logarithmic error regression loss.
    Scikit-implementation:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html
    Note: In order to avoid error, negative predictions are clipped to 0.
    This means that you should clip negative predictions manually after calling predict.
    """
    def __init__(self):
        self._name = "rmsle"
        self._maximize = False

    # mean_squared_log_error 함수로 계산하며, 음수 값을 0으로 클리핑 하여 RMSLE를 계산 
    def __call__(self, y_true, y_score):
        """
        Compute RMSLE of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSLE of predictions vs targets.
        """
        y_score = np.clip(y_score, a_min=0, a_max=None)
        return np.sqrt(mean_squared_log_error(y_true, y_score))


# 비지도 학습용 손실을 계산하는 메트릭 클래스를 정의
# _name은 "unsup_loss", 최적화는 False로 설정 
class UnsupervisedMetric(Metric):
    """
    Unsupervised metric
    """
    def __init__(self):
        self._name = "unsup_loss"
        self._maximize = False

    # 비지도 손실 함수를 사용해 손실 값을 계산 
    def __call__(self, y_pred, embedded_x, obf_vars):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_pred : torch.Tensor or np.array
            Reconstructed prediction (with embeddings)
        embedded_x : torch.Tensor
            Original input embedded by network
        obf_vars : torch.Tensor
            Binary mask for obfuscated variables.
            1 means the variables was obfuscated so reconstruction is based on this.

        Returns
        -------
        float
            MSE of predictions vs targets.
        """
        loss = UnsupervisedLoss(y_pred, embedded_x, obf_vars)
        return loss.item()


# 비지도 손실을 NumPy로 계산하는 메트릭 클래스를 정의
# _name은 "unsup_loss_numpy", 최적화는 False로 설정 
class UnsupervisedNumpyMetric(Metric):
    """
    Unsupervised metric
    """
    def __init__(self):
        self._name = "unsup_loss_numpy"
        self._maximize = False

    # UnsupervisedLossNumpy 함수를 사용해 손실을 계산 
    def __call__(self, y_pred, embedded_x, obf_vars):
        """
        Compute MSE (Mean Squared Error) of predictions.

        Parameters
        ----------
        y_pred : torch.Tensor or np.array
            Reconstructed prediction (with embeddings)
        embedded_x : torch.Tensor
            Original input embedded by network
        obf_vars : torch.Tensor
            Binary mask for obfuscated variables.
            1 means the variables was obfuscated so reconstruction is based on this.

        Returns
        -------
        float
            MSE of predictions vs targets.
        """
        return UnsupervisedLossNumpy(
            y_pred,
            embedded_x,
            obf_vars
        )


# 루트 평균 제곱 오차(RMSE)를 계산하는 메트릭 클래스를 정의
# _name은 "rmse", 최적화는 False로 설정
class RMSE(Metric):
    """
    Root Mean Squared Error.
    """
    def __init__(self):
        self._name = "rmse"
        self._maximize = False

    # 평균 제곱 오차를 계산한 후 그 값을 제곱근 하여 RMSE를 계산 
    def __call__(self, y_true, y_score):
        """
        Compute RMSE (Root Mean Squared Error) of predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Target matrix or vector
        y_score : np.ndarray
            Score matrix or vector

        Returns
        -------
        float
            RMSE of predictions vs targets.
        """
        return np.sqrt(mean_squared_error(y_true, y_score))


# 주어진 메트릭들이 올바르게 정의되었는지 확인하는 함수 
def check_metrics(metrics):
    """Check if custom metrics are provided.

    Parameters
    ----------
    metrics : list of str or classes
        List with built-in metrics (str) or custom metrics (classes).

    Returns
    -------
    val_metrics : list of str
        List of metric names.

    """
    
    # 빈 리스트를 생성 
    val_metrics = []
    
    # metrics 리스트를 순회하며, 문자열로 된 메트릭은 그대로 추가하고
    # Metric 클래스를 상속한 경우 그 이름을 리스트에 추가 
    for metric in metrics:
        if isinstance(metric, str):
            val_metrics.append(metric)
        elif issubclass(metric, Metric):
            val_metrics.append(metric()._name)
            
        # metrics 리스트의 항목이 문자열도 아니고 Metric 클래스의 서브 클래스도 아닌 경우 TypeError를 발생시킴 
        else:
            raise TypeError("You need to provide a valid metric format")
        
    # 모든 검증이 완료된 후, 유효한 메트릭 이름들을 담고 있는 val_metrics 리스트를 반환 
    return val_metrics
