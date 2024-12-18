import time
import datetime
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Any
import warnings

# Callback은 추상 기반 클래스로, 새로운 콜백을 생성할 때 사용할 수 있는 기반이 되는 클래스
# 콜백은 모델 학습 중 발생하는 다양한 이벤트에 개입할 수 있도록 설계된 클래스 
class Callback:
    """
    Abstract base class used to build new callbacks.
    """

    # 생성자 메서드로, 기본적으로 아무런 초기화도 수행하지 않음
    # 구체적인 콜백 클래스에서 필요한 초기화는 이 생성자를 재정의하여 수행하게 됨
    def __init__(self):
        pass

    # 콜백이 사용할 학습 파라미터를 설정하는 역할
    # params는 학습에 필요한 여러 설정이나 정보를 담고 있는 딕셔너리일 수 있으며, 이 값을 self.params에 저장
    def set_params(self, params):
        self.params = params

    # 콜백이 연동할 트레이너(모델)을 설정
    # model을 받아 self.trainer에 저장, 모델의 상태를 참조하거나 수정할 수 있도록함
    def set_trainer(self, model):
        self.trainer = model

    # 각 epoch의 시작시 호출 epoch는 현재 epoch 번호를 의미
    # logs는 epoch 시작 시 추가적인 정보를 전달할 수 있는 딕셔너리
    # 기본적으로는 아무 동작도 하지 않으며, 상속받은 클래스에서 이를 재정의하여 사용할 수 있음 
    def on_epoch_begin(self, epoch, logs=None):
        pass

    # 이 메서드는 각 epoch의 끝에서 호출
    # epoch와 logs를 인자로 받으며, on_epoch_begin과 마찬가지로 기본 동작은 없고, 재정의 해서 사용가능
    def on_epoch_end(self, epoch, logs=None):
        pass

    # 이 메서드는 각 batch의 시작시 호출
    # batch는 현재 batch의 인덱스를 나타내며 logs는 추가적인 정보를 전달할 수 있는 딕셔너리
    # 마찬가지로 기본 동작은 없고, 구체적인 콜백에서 재정의할 수 있음 
    def on_batch_begin(self, batch, logs=None):
        pass

    # 이 메서드는 각 batch의 끝에서 호출
    # batch와 logs를 인자로 받으며, batch 끝에서 필요한 작업을 수행할 수 있도록 재정의 할 수 있음 
    def on_batch_end(self, batch, logs=None):
        pass

    # 이 메서드는 전체 훈련이 시작될 때 호출됨
    # logs는 훈련이 시작될 때 전달할 수 있는 추가 정보를 담고 있음
    # 훈련 시작시 필요한 작업을 정의하기 위해 재정의 할 수 있음 
    def on_train_begin(self, logs=None):
        pass

    # 이 메서드는 훈련이 종료될 때 호출됨
    # logs는 훈련 종료 시의 정보를 담을 수 있음
    # 훈련이 끝난 후 필요한 작업을 정의할 수 있도록 재정의하여 사용할 수 있음
    def on_train_end(self, logs=None):
        pass


# @dataclass 데코레이터는 자동으로 클래스의 생성자와 기타 메서드를 생성해줌
# 이 클래스는 콜백 리스트를 관리하는 컨테이너 역할을 함
# CallbackContainer는 여러개의 콜백을 모아서 관리하며, 학습 도중에 발생하는 다양한 이벤트를 콜백들에게 전달
@dataclass
class CallbackContainer:
    """
    Container holding a list of callbacks.
    """

    # callbacks는 Callback 객체들의 리스트로, 기본적으로 빈 리스트로 초기화됨
    # field(default_factory=list) 는 리스트를 기본값으로 설정하는 데 사용
    callbacks: List[Callback] = field(default_factory=list)

    # apeend 메서드는 callback 객체를 callbacks 리스트에 추가하는 역할을 함
    # 콜백 리스트에 새로운 콜백을 추가할 때 사용
    def append(self, callback):
        self.callbacks.append(callback)

    # set_params 메서드는 각 콜백에 학습 파라미터를 설정함
    # 모든 콜백에 대해 set_params 메서드를 호출하여 동일한 학습 파라미터를 설정함
    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    # set_trainer 메서드는 콜백들이 사용할 트레이너(모델)를 설정
    # trainer 객체를 self.trainer에 저장하고, 모든 콜백에 대해 set_trainer 메서드를 호출해 트레이너 설정
    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    # on_epoch_begin 메서드는 각 epoch의 시작 시 호출됨
    # logs가 없을 경우 빈 딕셔너리로 초기화한 뒤, 모든 콜백에 대해 on_epoch_begin 메서드를 호출하여 epoch 시작 시 콜백이 동작하도록 함
    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    # on_epoch_end 메서드는 각 epoch의 끝에서 호출됨
    # logs 가 없을 경우 빈 딕셔너리로 초기화한 후, 모든 콜백에 대해 on_epoch_end 메서드를 호출
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    # on_batch_begin 메서드는 각 batch의 시작 시 호출됨
    # logs가 없을 경우 빈 딕셔너리로 초기화하고, 각 콜백의 on_batch_begin 메서드를 호출함
    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    # on_batch_end 메서드는 각 batch의 끝에서 호출
    # logs가 없을 경우 빈 딕셔너리로 초기화하고, 모든 콜백의 on_batch_end 메서드를 호출
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    # on_train_begin 메서드는 전체 훈련이 시작될 때 호출됨
    # logs 가 없을 경우 빈 딕셔너리로 초기화하고
    # 훈련 시작 시간을 기록한 후 모든 콜백의 on_train_begin 메서드를 호출함
    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs["start_time"] = time.time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    # on_train_end 메서드는 전체 훈련이 종료될 때 호출됨
    # logs 가 없을 경우 빈 딕셔너리로 초기화하고
    # 모든 콜백의 on_train_end 메서드를 호출하여 훈련 종료 시 콜백이 동작하도록 함
    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


# @dataclass 데코레이터를 사용하여 EarlyStopping 클래스를 저으이
# 이 클래스는 모델 학습을 중단하는 기능을 가진 콜백
# Callback 클래스를 상속받았으며, early_stopping_metric이 개선되지 않으면 학습을 종료 하는 역할 
@dataclass
class EarlyStopping(Callback):
    """EarlyStopping callback to exit the training loop if early_stopping_metric
    does not improve by a certain amount for a certain
    number of epochs.

    Parameters
    ---------
    early_stopping_metric : str
        Early stopping metric name 
    is_maximize : bool
        Whether to maximize or not early_stopping_metric
    tol : float
        minimum change in monitored value to qualify as improvement.
        This number should be positive.
    patience : integer
        number of epochs to wait for improvement before terminating.
        the counter be reset after each improvement

    """
    # 조기 종료를 결정하는 기준이 되는 지표 이름
    early_stopping_metric: str
    
    # 해당 지표를 최대화할지 여부를 나타냄 (최대화 해야하면 True, 최소화 해야하면 False)
    is_maximize: bool
    
    # 성능이 개선되었다고 간주할 변화량 이 값은 양수여야 함
    tol: float = 0.0
    
    # 성능 개선이 없을 때 몇 번의 epoch을 기다렸다가 학습을 중단할지 설정
    patience: int = 5

    # dataclass 에서 자동으로 생성되는 생성자 이후에 호출되는 메서드
    # 객체가 초기화된 후 추가적인 초기화 작업을 수행 
    def __post_init__(self):
        
        # 가장 좋은 성능을 기록한 epoch를 저장하는 변수
        self.best_epoch = 0
        
        # 조기 종료가 발생한 epoch를 저장하는 변수 
        self.stopped_epoch = 0
        
        # 성능 개선이 없었던 epoch의 수를 세는 변수 
        self.wait = 0
        
        # 가장 좋은 성능을 기록한 시점의 모델 가중치를 저장
        self.best_weights = None
        
        # 가장 좋은 성능을 기록한 시점의 성능 지표(손실값)로, 초기값은 무한대
        self.best_loss = np.inf
        
        # is_maximize가 True인 경우, best_loss를 최대화해야 하므로 best_loss의 부호를 바꿈 
        if self.is_maximize:
            self.best_loss = -self.best_loss
            
        # 부모 클래스인 Callback의 __init__메서드를 호출하여, 기본 초기화 작업을 수행
        super().__init__()

    # on_epoch_enc: 각 epoch의 끝에서 호출됨, 현재 epoch의 성능 지표를 가져옴
    # logs 에서 early_stopping_metric에 해당하는 값을 가져와 current_loss에 저장
    # 만약 current_loss가 None이면, 해당 epoch에서 처리할 사항이 없으므로 메서드를 종료
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return

        # 현재 손실과 가장 좋은 손실 간의 차이를 계산 
        loss_change = current_loss - self.best_loss
        
        # 최대화할 경우 현재 손실이 tol보다 많이 개선되었는지를 체크
        max_improved = self.is_maximize and loss_change > self.tol
        
        # 최소화할 경우 현재 손실이 tol보다 많이 개선되었는지를 체크
        min_improved = (not self.is_maximize) and (-loss_change > self.tol)
        
        # 만약 성능이 개선된 경우(max_improved 또는 min_improved), best_loss를 현재 손실로 업데이터, best_epoch를 현재 epoch로 설정
        # wait을 1로 리셋하고, best_weights에 현재 모델의 가중치를 저장
        if max_improved or min_improved:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.wait = 1
            self.best_weights = copy.deepcopy(self.trainer.network.state_dict())
            
        # 성능이 개선되지 않은 경우, wait을 증가시키고, wait이 parience보다 크거나 같아지면 조기 종료를 설정
        # stopped_epoch에 현재 epoch을 기록하고 self.trainer._stop_training을 True로 설정하여 학습을 종료
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer._stop_training = True
            self.wait += 1

    # on_train_end : 전체 학습이 종료될 때 호출
    # trainer의 best_epoch와 best_cost를 각각 best_epoch와 best_loss로 설정하여 학습 중 가장 좋은 성능을 기록한 epoch와 손실을 기록
    def on_train_end(self, logs=None):
        self.trainer.best_epoch = self.best_epoch
        self.trainer.best_cost = self.best_loss

        # best_weights가 존재하는 경우, 모델의 가중치를 best_weights로 로드하여 학습 중 가장 좋은 성능의 가중치를 복원 
        if self.best_weights is not None:
            self.trainer.network.load_state_dict(self.best_weights)

        # 조기 종료가 발생한 경우 (stopped_epoch > 0),
        # 조기 종료가 발생한 epoch와 최상의 성능을 기록한 epoch, 성능 지표 값을 출력
        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occurred at epoch {self.stopped_epoch}"
            msg += (
                f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            print(msg)
            
        # 조기 종료가 발생하지 않은 경우, max_epochs에 도달하여 학습을 종료했음을 알리는 메시지 출력
        else:
            msg = (
                f"Stop training because you reached max_epochs = {self.trainer.max_epochs}"
                + f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            print(msg)
            
        # 학습이 종료되면서 가장 좋은 epoch의 가중치가 자동으로 사용된다는 경고 메시지를 출력
        wrn_msg = "Best weights from best epoch are automatically used!"
        warnings.warn(wrn_msg)

# @dataclass 데코레이터를 사용하여 History 클래스를 정의 
# 이 클래스는 학습 중 이벤트를 기록하는 콜백
@dataclass
class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every SuperModule.

    Parameters
    ---------
    trainer : DeepRecoModel
        Model class to train
    verbose : int
        Print results every verbose iteration

    """

    # trainer: 모델 학습에 사용하는 트레이너 객체를 설정
    # verbose: 결과를 출력할 빈도, 디폴트는 1
    trainer: Any
    verbose: int = 1

    # dataclass 에서 자동 생성되는 생성자 이후에 호출
    def __post_init__(self):
        
        # super().__init__()을 호출하여 부모 클래스인 Callback의 생성자를 호출
        super().__init__()
        
        # 학습 중 처리한 샘플 수를 기록하는 변수 
        self.samples_seen = 0.0
        
        # 학습 시작 이후 경과 시간을 기록하는 변수
        self.total_time = 0.0

    # 학습이 시작될 때 호출됨
    def on_train_begin(self, logs=None):
        
        # 기록할 데이터를 저장할 딕셔너리를 초기화
        # 기본적으로 loss,lr,및 트레이너의 메트릭 이름에 대한 빈 리스트를 포함함
        self.history = {"loss": []}
        self.history.update({"lr": []})
        self.history.update({name: [] for name in self.trainer._metrics_names})
        
        # 학습 시작 시간을 저장 
        self.start_time = logs["start_time"]
        
        # 현재 epoch의 손실값을 저장할 변수를 초기화 
        self.epoch_loss = 0.0

    # 각 epoch가 시작될 때 호출됨
    def on_epoch_begin(self, epoch, logs=None):
        
        # 현재 epoch의 메트릭을 저장할 딕셔너리를 초기화함 기본적으로 loss만 포함
        self.epoch_metrics = {"loss": 0.0}
        
        # 현재 epoch에서 처리한 샘플 수를 저장할 변수를 초기화 
        self.samples_seen = 0.0

    # 각 epoch가 끝날 때 호출됨
    def on_epoch_end(self, epoch, logs=None):
        
        # 현재 epoch의 손실값을 기록, 각 메트릭을 self.history에 추가하여 기록
        self.epoch_metrics["loss"] = self.epoch_loss
        for metric_name, metric_value in self.epoch_metrics.items():
            self.history[metric_name].append(metric_value)
            
        # verbose가 0일 경우 메시지를 출력하지 않음
        # epochrk verbose의 배수가 아닐 경우에도 메시지를 출력하지 않음
        # 현재 epoch와 각 메트릭 값을 포함한 메시지를 생성하고 출력함
        # 학습 시작 이후 경과 시간을 계산하여 출력함
        if self.verbose == 0:
            return
        if epoch % self.verbose != 0:
            return
        msg = f"epoch {epoch:<3}"
        for metric_name, metric_value in self.epoch_metrics.items():
            if metric_name != "lr":
                msg += f"| {metric_name:<3}: {np.round(metric_value, 5):<8}"
        self.total_time = int(time.time() - self.start_time)
        msg += f"|  {str(datetime.timedelta(seconds=self.total_time)) + 's':<6}"
        print(msg)

    # on_batch_end: 각 배치가 끝날 때 호출됨
    # batch_size를 가져와 현재 배치의 손실값을 포함하여 self.epoch_loss를 업데이트
    # samples_seen에 현재 배치의 샘플 수를 더함 
    def on_batch_end(self, batch, logs=None):
        batch_size = logs["batch_size"]
        self.epoch_loss = (
            self.samples_seen * self.epoch_loss + batch_size * logs["loss"]
        ) / (self.samples_seen + batch_size)
        self.samples_seen += batch_size

    # History 객체에서 특정 이름의 기록을 가져오는 메서드 
    def __getitem__(self, name):
        return self.history[name]
    
    # History 객체를 문자열로 표현하는 메서드 self.history를 문자열로 반환
    def __repr__(self):
        return str(self.history)

    def __str__(self):
        return str(self.history)


# LRSchedulerCallback 클래스를 데이터 클래스 형태로 정의
@dataclass
class LRSchedulerCallback(Callback):
    """Wrapper for most torch scheduler functions.

    Parameters
    ---------
    scheduler_fn : torch.optim.lr_scheduler
        Torch scheduling class
    scheduler_params : dict
        Dictionnary containing all parameters for the scheduler_fn
    is_batch_level : bool (default = False)
        If set to False : lr updates will happen at every epoch
        If set to True : lr updates happen at every batch
        Set this to True for OneCycleLR for example
    """

    # PyTorch의 스케줄러 클래스를 지정
    # 예를 들어, torch.optim.lr_scheduler.stepLR 같은 스케줄러를 전달할 수 있음 
    scheduler_fn: Any
    
    # 학습률을 조정할 옵티마이저 지정 
    optimizer: Any
    
    # 스케줄러를 초기화하는 데 필요한 매개변수를 포함하는 딕셔너리
    scheduler_params: dict
    
    # 스케줄러를 업데이트할 때 사용할 메트릭의 이름
    early_stopping_metric: str
    
    # 학습률 업데이트를 배치 단위로 할지(epoch 단위로 할지) 결정하는 플래그 
    is_batch_level: bool = False

    # dataclass에서 생성자 호출 후 추가적인 초기화 작업을 수행
    def __post_init__(
        self,
    ):
        # 스케줄러 클래스가 메트릭 기반 업데이트를 지원하는지 확인
        # scheduler_fn에 "is_better" 속성이 있는지 체크
        self.is_metric_related = hasattr(self.scheduler_fn, "is_better")
        
        # 전달된 스케줄러 함수와 옵티마이저, 그리고 매개변수로 스케줄러 객체를 생성 
        self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        
        # 부모 클래스인 Callback의 생성자를 호출
        super().__init__()

    # 배치가 끝날 때 호출
    def on_batch_end(self, batch, logs=None):
        
        # True 인 경우, 배치가 끝날 때마다 스케줄러의 step() 메서드를 호출하여 학습률 업데이트 
        # False 인 경우, 이 메서드는 아무 작업도 수행하지 않음 
        if self.is_batch_level:
            self.scheduler.step()
        else:
            pass

    # epoch가 끝날 때 호출
    # logs에서 early_stopping_metric을 통해 현재 메트릭 값을 가져옴
    def on_epoch_end(self, epoch, logs=None):
        
        # None 인 경우, 아무 작업도 수행하지 않음 
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return
        
        # False 인 경우에만 이 메서드가 호출됨 
        if self.is_batch_level:
            pass
        else:
            # True 인 경우, 현재 메트릭 값 (current_loss)을 사용하여 스케줄러의 step() 메서드를 호출
            # False 인 경우, 단순히 step() 메서드를 호출하여 스케줄러를 업데이트 
            if self.is_metric_related:
                self.scheduler.step(current_loss)
            else:
                self.scheduler.step()
