import torch
import numpy as np
from scipy.special import softmax
from pytorch_tabnet.utils import SparsePredictDataset, PredictDataset, filter_weights
from pytorch_tabnet.abstract_model import TabModel
from pytorch_tabnet.multiclass_utils import infer_output_dim, check_output_dim
from torch.utils.data import DataLoader
import scipy


# TabNet 모델을 사용하여 분류 작업을 수행하는 클래스로 이 클래스는 TabModel 클래스르 상속받음 
class TabNetClassifier(TabModel):
    
    # __post__init__메서드는 분류 작업에 필요한 기본 설정을 정의
    # 이 메서드는 부모 클래스의 초기화를 먼저 수행하고
    # 이후 분류 작업에 맞게 손실 함수(cross_entropy)와 기본 평가 지표(accuracy)를 설정 
    def __post_init__(self):
        super(TabNetClassifier, self).__post_init__()
        self._task = 'classification'
        self._default_loss = torch.nn.functional.cross_entropy
        self._default_metric = 'accuracy'

    # 주어진 가중치 딕셔너리를 업데이트함
    # 이 함수는 분류 클래스에 따른 가중치를 매핑함
    def weight_updater(self, weights):
        """
        Updates weights dictionary according to target_mapper.

        Parameters
        ----------
        weights : bool or dict
            Given weights for balancing training.

        Returns
        -------
        bool or dict
            Same bool if weights are bool, updated dict otherwise.

        """
        
        # weights가 정수이면 그대로 반환하고, 딕셔너리이면 각 클래스에 맞는 값으로 변환 후 반환하며 다른 경우는 원본을 반환
        if isinstance(weights, int):
            return weights
        elif isinstance(weights, dict):
            return {self.target_mapper[key]: value for key, value in weights.items()}
        else:
            return weights

    # 타깃 데이터를 클래스 인덱스로 변환. target_mapper를 사용해 타깃 값을 매핑
    def prepare_target(self, y):
        return np.vectorize(self.target_mapper.get)(y)

    # 예측값과 실제값 사이의 손실을 계산, long()을 사용해 정수형으로 변환한 y_true를 입력으로 받음 
    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true.long())

    # 훈련 중에 필요한 파라미터를 업데이트. 출력 차원, 클래스, 가중치 등을 설정 
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights,
    ):
        
        # infer_output_dim을 통해 출력 차원과 훈련 레이블을 추론하고, 검증 세트와 일치하는지 확인
        # 출력 차원이 2일경우 auc를 기본 메트릭으로 설정하고, 그렇지 않으면 accuracy를 사용
        output_dim, train_labels = infer_output_dim(y_train)
        for X, y in eval_set:
            check_output_dim(train_labels, y)
        self.output_dim = output_dim
        self._default_metric = ('auc' if self.output_dim == 2 else 'accuracy')
        self.classes_ = train_labels
        
        # 클래스 레이블을 인덱스에 매핑하는 target_mapper와
        # 반대로 인덱스를 클래스 레이블로 매핑하는 preds_mapper를 설정, 또한 가중치를 업데이트 
        self.target_mapper = {
            class_label: index for index, class_label in enumerate(self.classes_)
        }
        self.preds_mapper = {
            str(index): class_label for index, class_label in enumerate(self.classes_)
        }
        self.updated_weights = self.weight_updater(weights)

    # 여러 배치에서의 실제값(y_true)과 예측 점수(y_score)를 하나로 합침
    # softmax를 적용해 예측값을 확률로 변환
    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.hstack(list_y_true)
        y_score = np.vstack(list_y_score)
        y_score = softmax(y_score, axis=1)  
        
        return y_true, y_score

    # 예측값을 반환. argmax로 가장 높은 확률을 가진 클래스를 선택
    # 해당 클래스를 preds_mapper로 변환
    def predict_func(self, outputs):
        outputs = np.argmax(outputs, axis=1)
        return np.vectorize(self.preds_mapper.get)(outputs.astype(str))

    # 입력 데이터에 대해 모델의 확률 예측을 수행
    # 먼저 네트워크를 평가 모드로 설정 
    def predict_proba(self, X):
        """
        Make predictions for classification on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        res : np.ndarray

        """
        self.network.eval()

        # 입력이 희소행렬인지 확인하고
        # 그에 맞는 데이터셋(SparsePredictDataset 또는 PredictDataset)을 사용해 데이터 로더를 설정 
        if scipy.sparse.issparse(X):
            dataloader = DataLoader(
                SparsePredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                PredictDataset(X),
                batch_size=self.batch_size,
                shuffle=False,
            )

        # 배치별로 데이터를 처리하고, 출력값에 softmax를 적용해 예측 확률을 계산한 후 결과를 반환
        results = []
        try:
            for batch_nb, data in enumerate(dataloader):
                data = data.to(self.device).float()

                output, M_loss = self.network(data)

                print(f"Output shape: {output.shape}")
                print(f"Output values: {output}")

                predictions = torch.nn.Softmax(dim=1)(output).cpu().detach().numpy()
                results.append(predictions)

        except Exception as e:
            print(f"An error occurred: {e}")
     
        res = np.vstack(results)
        return res
    


# 회귀 작업을 수행하는 모델로 TabModel을 상속받아 동작 
class TabNetRegressor(TabModel):
    
    # 회귀 작업을 위한 설정 수행
    # 기본 손실 함수로 mse_loss를, 기본 평가 지표로 mse를 사용 
    def __post_init__(self):
        super(TabNetRegressor, self).__post_init__()
        self._task = 'regression'
        self._default_loss = torch.nn.functional.mse_loss
        self._default_metric = 'mse'

    # 회귀 작업이므로 타깃 데이터를 그대로 반환 
    def prepare_target(self, y):
        return y


    # 예측값과 실제값을 사용해 손실을 계산 
    def compute_loss(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


    # 훈련 시 사용할 파라미터를 업데이트함
    # 타깃 데이터의 차원이 2D가 아닌 경우 오류를 발생시키고, 출력 차원을 설정, 가중치도 업데이트하고 필터링
    def update_fit_params(
        self,
        X_train,
        y_train,
        eval_set,
        weights
    ):
        if len(y_train.shape) != 2:
            msg = "Targets should be 2D : (n_samples, n_regression) " + \
                  f"but y_train.shape={y_train.shape} given.\n" + \
                  "Use reshape(-1, 1) for single regression."
            raise ValueError(msg)
        self.output_dim = y_train.shape[1]
        self.preds_mapper = None

        self.updated_weights = weights
        filter_weights(self.updated_weights)


    # 회귀 작업에서는 예측값을 그대로 반환
    def predict_func(self, outputs):
        return outputs


    # 여러 배치에서의 실제값과 예측값을 하나로 합쳐 반환
    def stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score
