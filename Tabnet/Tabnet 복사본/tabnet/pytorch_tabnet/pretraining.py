import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_tabnet import tab_network
from pytorch_tabnet.utils import (
    create_explain_matrix,
    filter_weights,
    SparsePredictDataset,
    PredictDataset,
    check_input,
    create_group_matrix,
)
from torch.nn.utils import clip_grad_norm_
from pytorch_tabnet.pretraining_utils import (
    create_dataloaders,
    validate_eval_set,
)
from pytorch_tabnet.metrics import (
    UnsupMetricContainer,
    check_metrics,
    UnsupervisedLoss,
)
from pytorch_tabnet.abstract_model import TabModel
import scipy


class TabNetPretrainer(TabModel):
    
    # 객체 초기화 후 추가 설정을 수행
    # 부모 클래스의 __post_init__메서드를 호출
    # self._task를 'unsupervised'로 설정
    # 기본 손실함수를 UnsupervisedLoss로 설정
    # 기본 메트릭을 'unsup_loss_numpy로 설정 
    def __post_init__(self):
        super(TabNetPretrainer, self).__post_init__()
        self._task = 'unsupervised'
        self._default_loss = UnsupervisedLoss
        self._default_metric = 'unsup_loss_numpy'

    # 타켓 데이터를 준비
    # 사전 훈련에서는 타겟을 그대로 반환, 이 메서드는 구체적인 변환을 필요로 하지 않음
    def prepare_target(self, y):
        return y

    # 손실 값을 계산
    # self.loss_fn에 정의된 손실 함수를 사용하여 손실값을 계산
    # output, embedded_x, obf_vars를 손실 함수에 전달 
    def compute_loss(self, output, embedded_x, obf_vars):
        return self.loss_fn(output, embedded_x, obf_vars)

    # 훈련 파라미터를 업데이트
    # weights를 self.updated_weights에 저장
    # filter_weights 함수를 사용하여 가중치를 필터링
    # self.preds_mapper를 None으로 설정
    def update_fit_params(
        self,
        weights,
    ):
        self.updated_weights = weights
        filter_weights(self.updated_weights)
        self.preds_mapper = None

    # 모델을 훈련시킴
    # 여러 파라미터를 설정하여 훈련을 준비
    # loss_fn이 제공되지 않으면 기본 손실 함수를 사용
    # check_input 함수를 사용하여 입력 데이터를 검증
    # update_fit_params를 호출하여 훈련 파라미터를 업데이트
    # 평가 세트를 검증하고, 훈련 및 검증 데이터 로더를 생성
    # 네트워크가 초기화되지 않았거나 warm_start가 False인 경우, _set_network를 호출
    # 네트워크 매개변수를 업데이트하고, 메트릭, 옵디마이저, 콜백을 설정
    # 에포크 반복을 통해 훈련을 수행하고, 각 에포크의 시작과 끝에 콜백 메서드를 호출
    # 조기 종료 조건이 만족되면 훈련을 중지
    # 훈련이 끝난 후 네트워크를 평가 모드로 설정 
    def fit(
        self,
        X_train,
        eval_set=None,
        eval_name=None,
        loss_fn=None,
        pretraining_ratio=0.5,
        weights=0,
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=True,
        callbacks=None,
        pin_memory=True,
        warm_start=False
    ):
        """Train a neural network stored in self.network
        Using train_dataloader for training data and
        valid_dataloader for validation.

        Parameters
        ----------
        X_train : np.ndarray
            Train set to reconstruct in self supervision
        eval_set : list of np.array
            List of evaluation set
            The last one is used for early stopping
        eval_name : list of str
            List of eval set names.
        eval_metric : list of str
            List of evaluation metrics.
            The last metric is used for early stopping.
        loss_fn : callable or None
            a PyTorch loss function
            should be left to None for self supervised and non experts
        pretraining_ratio : float
            Between 0 and 1, percentage of feature to mask for reconstruction
        weights : np.array
            Sampling weights for each example.
        max_epochs : int
            Maximum number of epochs during training
        patience : int
            Number of consecutive non improving epoch before early stopping
        batch_size : int
            Training batch size
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization (virtual_batch_size < batch_size)
        num_workers : int
            Number of workers used in torch.utils.data.DataLoader
        drop_last : bool
            Whether to drop last batch during training
        callbacks : list of callback function
            List of custom callbacks
        pin_memory: bool
            Whether to set pin_memory to True or False during training
        """
        # update model name

        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.input_dim = X_train.shape[1]
        self._stop_training = False
        self.pin_memory = pin_memory and (self.device.type != "cpu")
        self.pretraining_ratio = pretraining_ratio
        eval_set = eval_set if eval_set else []

        if loss_fn is None:
            self.loss_fn = self._default_loss
        else:
            self.loss_fn = loss_fn

        check_input(X_train)

        self.update_fit_params(
            weights,
        )

        # Validate and reformat eval set depending on training data
        eval_names = validate_eval_set(eval_set, eval_name, X_train)
        train_dataloader, valid_dataloaders = self._construct_loaders(
            X_train, eval_set
        )

        if not hasattr(self, "network") or not warm_start:
            # model has never been fitted before of warm_start is False
            self._set_network()

        self._update_network_params()
        self._set_metrics(eval_names)
        self._set_optimizer()
        self._set_callbacks(callbacks)

        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()

        # Training loop over epochs
        for epoch_idx in range(self.max_epochs):

            # Call method on_epoch_begin for all callbacks
            self._callback_container.on_epoch_begin(epoch_idx)

            self._train_epoch(train_dataloader)

            # Apply predict epoch to all eval sets
            for eval_name, valid_dataloader in zip(eval_names, valid_dataloaders):
                self._predict_epoch(eval_name, valid_dataloader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(
                epoch_idx, logs=self.history.epoch_metrics
            )

            if self._stop_training:
                break

        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self.network.eval()

    # 네트워크 및 설명 행렬을 설정
    # pretraining_ratio가 설정되지 않은 경우 기본값 0.5를 사용
    # 난수 시드를 설정하여 재현성을 보장
    # create_group_matrix를 사용하여 그룹 행렬을 생성
    # TabNetPretraining 네트워크를 설정하고 장치로 이동시킴
    # create_explain_matrix를 사용하여 설명 행렬을 생성 
    def _set_network(self):
        """Setup the network and explain matrix."""
        if not hasattr(self, 'pretraining_ratio'):
            self.pretraining_ratio = 0.5
        torch.manual_seed(self.seed)

        self.group_matrix = create_group_matrix(self.grouped_features, self.input_dim)

        self.network = tab_network.TabNetPretraining(
            self.input_dim,
            pretraining_ratio=self.pretraining_ratio,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=1,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            n_shared_decoder=self.n_shared_decoder,
            n_indep_decoder=self.n_indep_decoder,
            epsilon=self.epsilon,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type,
            group_attention_matrix=self.group_matrix.to(self.device),
        ).to(self.device)

        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )

    # 네트워크의 매개변수를 업데이트함
    # 네트워크의 virtual_batch_size와 pretraining_ratio를 업데이트된 값으로 설정
    def _update_network_params(self):
        self.network.virtual_batch_size = self.virtual_batch_size
        self.network.pretraining_ratio = self.pretraining_ratio

    # 평가 메트릭을 설정
    # metrics 리스트를 기본 메트릭으로 초기화
    # check_metrics를 호출하여 메트릭을 검증
    # eval_names에 대한 메트릭 컨테이너를 설정
    # 각 평가 세트에 대해 UnsupMetricContainer를 생성하고 _metric_container_dict에 추가
    # _metrics와 _metrics_names 리스트를 업데이트
    # 조기 종료에 사용할 메트릭을 설정(early_stopping_metric)
    def _set_metrics(self, eval_names):
        """Set attributes relative to the metrics.

        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.

        """
        metrics = [self._default_metric]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update(
                {name: UnsupMetricContainer(metrics, prefix=f"{name}_")}
            )

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (
            self._metrics_names[-1] if len(self._metrics_names) > 0 else None
        )

    # 훈련 및 평가 데이터 로더를 생성
    # create_dataloaders를 호출하여 훈련 및 검증 데이터 로더를 생성
    # 훈련 데이터와 평가 데이터 세트를 사용하여 train_dataloader와 valid_dataloaders를 반환
    def _construct_loaders(self, X_train, eval_set):
        """Generate dataloaders for unsupervised train and eval set.

        Parameters
        ----------
        X_train : np.array
            Train set.
        eval_set : list of tuple
            List of eval tuple set (X, y).

        Returns
        -------
        train_dataloader : `torch.utils.data.Dataloader`
            Training dataloader.
        valid_dataloaders : list of `torch.utils.data.Dataloader`
            List of validation dataloaders.

        """
        train_dataloader, valid_dataloaders = create_dataloaders(
            X_train,
            eval_set,
            self.updated_weights,
            self.batch_size,
            self.num_workers,
            self.drop_last,
            self.pin_memory,
        )
        return train_dataloader, valid_dataloaders

    # 네트워크의 한 에포크를 훈련
    # 네트워크를 훈련 모드로 설정
    # train_loader에서 배치별로 데이터를 가져와 _train_batch를 호출
    # 배치의 시작과 끝에 콜백을 호출
    # 에포크 후 학습률을 기록하고 epoch_metrics를 업데이트 
    def _train_epoch(self, train_loader):
        """
        Trains one epoch of the network in self.network

        Parameters
        ----------
        train_loader : a :class: `torch.utils.data.Dataloader`
            DataLoader with train set
        """
        self.network.train()

        for batch_idx, X in enumerate(train_loader):
            self._callback_container.on_batch_begin(batch_idx)

            batch_logs = self._train_batch(X)

            self._callback_container.on_batch_end(batch_idx, batch_logs)

        epoch_logs = {"lr": self._optimizer.param_groups[-1]["lr"]}
        self.history.epoch_metrics.update(epoch_logs)

        return

    # 배치 단위로 데이터를 훈련
    # 배치 크기를 기록
    # 데이터를 장치로 이동 시키고 float형으로 변환
    # 네트워크의 모든 파라미터의 기울기를 초기화
    # 네트워크의 출력을 얻고 손실을 계산
    # 역전파와 최적화를 수행(기울기 클리핑을 포함)
    # 손실을 기록 
    def _train_batch(self, X):
        """
        Trains one batch of data

        Parameters
        ----------
        X : torch.Tensor
            Train matrix

        Returns
        -------
        batch_outs : dict
            Dictionnary with "y": target and "score": prediction scores.
        batch_logs : dict
            Dictionnary with "batch_size" and "loss".
        """
        batch_logs = {"batch_size": X.shape[0]}

        X = X.to(self.device).float()

        for param in self.network.parameters():
            param.grad = None

        output, embedded_x, obf_vars = self.network(X)
        loss = self.compute_loss(output, embedded_x, obf_vars)

        # Perform backward pass and optimization
        loss.backward()
        if self.clip_value:
            clip_grad_norm_(self.network.parameters(), self.clip_value)
        self._optimizer.step()

        batch_logs["loss"] = loss.cpu().detach().numpy().item()

        return batch_logs

    # 검증 세트에 대해 예측을 수행하고 메트릭을 업데이트
    # 네트워크를 평가 모드로 설정
    # 배치별로 예측을 수행하여 결과를 리스트에 저장
    # 리스트를 스택하여 최종 출력을 생성
    # 메트릭 컨테이너를 사용하여 메트릭을 계산하고 기록
    # 네트워크를 다시 훈련 모드로 설정
    def _predict_epoch(self, name, loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self.network.eval()

        list_output = []
        list_embedded_x = []
        list_obfuscation = []
        # Main loop
        for batch_idx, X in enumerate(loader):
            output, embedded_x, obf_vars = self._predict_batch(X)
            list_output.append(output.cpu().detach().numpy())
            list_embedded_x.append(embedded_x.cpu().detach().numpy())
            list_obfuscation.append(obf_vars.cpu().detach().numpy())

        output, embedded_x, obf_vars = self.stack_batches(list_output,
                                                          list_embedded_x,
                                                          list_obfuscation)

        metrics_logs = self._metric_container_dict[name](output, embedded_x, obf_vars)
        self.network.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    # 단일 배치에 대해 예측을 수행
    # 데이터를 장치로 이동시키고 float형으로 변환
    # 네트워크를 통해 예측을 수행
    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = X.to(self.device).float()
        return self.network(X)

    # 배치별 출력을 스택하여 전체 출력을 생성
    # list_output, list_embedded_x, list_obfuscation을 수직으로 스택하여 최종 결과를 생성 
    def stack_batches(self, list_output, list_embedded_x, list_obfuscation):
        output = np.vstack(list_output)
        embedded_x = np.vstack(list_embedded_x)
        obf_vars = np.vstack(list_obfuscation)
        return output, embedded_x, obf_vars

    # 주어진 데이터에 대해 예측을 수행
    # 네트워크를 평가 모드로 설정
    # 입력 데이터가 희소 행렬인지 확인하고 적절한 데이터 로더를 생성
    # 데이터 배치별로 예측을 수행하여 결과를 리스트에 저장
    # 리스트를 스택하여 최종 예측 결과를 반환 
    def predict(self, X):
        """
        Make predictions on a batch (valid)

        Parameters
        ----------
        X : a :tensor: `torch.Tensor` or matrix: `scipy.sparse.csr_matrix`
            Input data

        Returns
        -------
        predictions : np.array
            Predictions of the regression problem
        """
        self.network.eval()

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

        results = []
        embedded_res = []
        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()
            output, embeded_x, _ = self.network(data)
            predictions = output.cpu().detach().numpy()
            results.append(predictions)
            embedded_res.append(embeded_x.cpu().detach().numpy())
        res_output = np.vstack(results)
        embedded_inputs = np.vstack(embedded_res)
        return res_output, embedded_inputs
