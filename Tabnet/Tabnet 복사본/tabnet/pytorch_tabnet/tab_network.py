import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from pytorch_tabnet import sparsemax


# GLU(Gated Linear Unit)가 아닌 모듈을 초기화한다
# gain_value: 입력과 출력 차원을 기반으로 gain_value를 계산하여 초기화를 위한 파라미터로 사용됨
# torch.nn.init.xavier_normal_: Xavier 초기화를 사용해 module.weight를 gain_value로 초기화 
def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


# GLU 모듈을 초기화 
# gain_value: GLU의 경우 입력 차원을 기반으로 계산됨
# torch.nn.init.xavier_normal_: Xavier 초기화를 사용해 GLU 모듈의 가중치를 초기화
def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


# GBN(Ghost Batch Normalization)을 구현하는 클래스
# 작은 배치 크기로 배치 정규화를 수행하며 큰 배치에서 발생할 수 있는 불안정성을 줄임
class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    # BatchNorm1d: 1차원 배치 정규화를 적용하는 모듈을 생성 
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    # forward: 입력 데이터를 작은 덩어리로 나누어 각 덩어리마다 배치 정규화를 적용한 후, 결과를 다시 결합
    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


# TabNet의 핵심 부분을 정의하는 클래스로, 임베딩 레이어는 포함되지 않음 
class TabNetEncoder(torch.nn.Module):
    def __init__(
        self,
        input_dim=525,   # 입력차원의 크기
        output_dim=16,  # 네트워크의 출력차원, 예를들어, 이진 분류는 2, 회귀는 1
        n_d=8,   # 예측 레이어의 차원
        n_a=8,   # attention 레이어의 차원
        n_steps=3,   # 네트워크의 단계 수
        gamma=1.3,   # 주의 업데이트를 위한 스케일링 계수
        n_independent=2,   # 각 GLU 블록에 독립적인 GLU 레이어의 수
        n_shared=2,   # 각 GLU 블록에 공유되는 GLU 레이어의 수
        epsilon=1e-15,   # 작은 값으로, 로그 계산에서 0을 방지 
        virtual_batch_size=128,   # GBN을 위한 배치 크기 
        momentum=0.02,   # 배치 정규화에서 사용하는 모멘텀 값
        mask_type="sparsemax",   # sparsemax or entmax와 같은 마스킹 함수
        group_attention_matrix=None,   # 그룹화된 주의도를 위한 행렬 
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        
        super(TabNetEncoder, self).__init__()   # torch.nn.Module의 초기화 함수를 호출
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)   # output_dim이 리스트인지 확인하여 멀티태스크인지 판별 
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)   # 초기입력에 배치 정규화 적용
        self.group_attention_matrix = group_attention_matrix

        # 그룹 주의도 행렬이 주어지지 않으면, input_dim 크기의 단위 행렬을 생성 
        # 그렇지 않으면, 주어진 그룹 주의도 행렬의 크기를 사용
        if self.group_attention_matrix is None:
            # no groups
            self.group_attention_matrix = torch.eye(self.input_dim)
            self.attention_dim = self.input_dim
        else:
            self.attention_dim = self.group_attention_matrix.shape[0]

        # 공유 GLU 블록: 공유할 GLU 레이어가 존재하는 경우, n_shared 만큼 Linear 레이어를 추가 
        # 첫번째 Linear 레이어는 입력 차원과 예측 및 주의 레이어 차원에 따라 정의됨
        # 이후 레이어는 n_d + n_a 크기를 갖음
        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(
                        Linear(self.input_dim, 2 * (n_d + n_a), bias=False)
                    )
                else:
                    shared_feat_transform.append(
                        Linear(n_d + n_a, 2 * (n_d + n_a), bias=False)
                    )

        # 공유할 GLU 레이어가 없는 경우, shared_feat_transform을 None으로 설정 
        else:
            shared_feat_transform = None

        # 초기 분할기: FeatTransformer를 사용해 입력을 n_d + n_a 크기로 환산
        self.initial_splitter = FeatTransformer(
            self.input_dim,
            n_d + n_a,
            shared_feat_transform,
            n_glu_independent=self.n_independent,
            virtual_batch_size=self.virtual_batch_size,
            momentum=momentum,
        )

        # 특징 변환기(feat_transformers)와 주의 변환기(att_transformers)를 저장할 리스트 생성 
        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        # 각 단계마다 FeatTransformer와 AttentiveTransformer를 생성하여 리스트에 추가
        # FeatTransformer: 입력을 변환하여 예측 및 주의 레이어에 적합한 차원으로 변환
        # AttentiveTransformer: 입력에 대한 주의도를 계산 
        for step in range(n_steps):
            transformer = FeatTransformer(
                self.input_dim,
                n_d + n_a,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            attention = AttentiveTransformer(
                n_a,
                self.attention_dim,
                group_matrix=group_attention_matrix,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
                mask_type=self.mask_type,
            )
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)


    # 입력 데이터 정규화: 입력 데이터 X에 대해 배치 정규화를 먼저 적용
    def forward(self, x, prior=None):
        x = self.initial_bn(x)

        # 배치 크기: 입력 데이터의 배치 크기를 저장
        # prior 초기화: 이전 단계에서의 주의도를 추적하는 prior를 설정, 초기값은 1
        bs = x.shape[0]  # batch size
        if prior is None:
            prior = torch.ones((bs, self.attention_dim)).to(x.device)

        # 손실 초기화: 마스크 손실(M_loss)을 0으로 초기화함
        # 특징 변환: 초기 분할기(initial_splitter)를 통해 주의(attention)와 예측 정보를 분리, 주의 레이어만 사용
        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d :]
        
        # 각 단계의 출력 저장: 여러 단계의 출력 값을 저장할 리스트를 초기화 
        # 주의 계산: att_transformers에서 prior와 att를 사용해 주의 마스크(M)를 계산 
        steps_output = []
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            
            # 손실 업데이트: 마스크 손실을 갱신, 이는 주의 마스크 M에 로그를 취한 후, 평균을 계산하여 누적
            # cross entropy 수식 코드로 표현
            M_loss += torch.mean(
                torch.sum(torch.mul(M, torch.log(M + self.epsilon)), dim=1)
            )
            
            # update prior
            # prior 업데이트: 현재 마스크 M을 사용하여 prior를 갱신 
            prior = torch.mul(self.gamma - M, prior)
            
            # output
            # 마스크된 입력 계산: M과 그룹 주의도 행렬을 곱하여 입력 데이터 X에 마스킹을 적용한 결과를 계산 
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            masked_x = torch.mul(M_feature_level, x)
            
            # 특징 변환 및 출력 저장: 마스크된 입력을 변환하여 출력(d)를 얻고, 이를 활성화 함수 ReLU()로 정규화한 후 저장 
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, : self.n_d])
            steps_output.append(d)
            
            # update attention
            # 다음 주의 정보 업데이트: 예측 레이어 이외의 부분을 att로 업데이트하여 다음 단계에서 사용할 준비를 함
            att = out[:, self.n_d :]

        # 손실 평균: n_steps에 따라 마스크 손실을 평균화하고, 모든 단계의 출력과 함께 반환
        M_loss /= self.n_steps
        return steps_output, M_loss

    # 입력 정규화: 입력 데이터에 배치 정규화를 적용
    def forward_masks(self, x):
        x = self.initial_bn(x)
        
        # prior 초기화: forward와 유사하게 Prior를 1로 초기화함
        # 마스크 설명 변수: 입력 크기와 동일한 크기의 M_explain을 0으로 초기화하여 주의 설명 정보를 저장할 공간을 마련
        bs = x.shape[0]  # batch size
        prior = torch.ones((bs, self.attention_dim)).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        
        # 특징 변환: 초기 분할기를 통해 att를 얻고, 각 단계의 마스크를 저장할 딕셔너리를 생성 
        att = self.initial_splitter(x)[:, self.n_d :]
        masks = {}

        # 주의 마스크 계산 및 저장: 각 단계마다 주의 마스크 M을 계산하고, 이를 masks dictionary에 저장 
        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            M_feature_level = torch.matmul(M, self.group_attention_matrix)
            masks[step] = M_feature_level
            
            # update prior
            # prior 업데이트 및 마스크된 입력 변환: 마스크된 입력을 feat_transformers로 변환한 후 활성화 함수를 적용
            prior = torch.mul(self.gamma - M, prior)
            
            # output
            # mask * X (element wise)
            masked_x = torch.mul(M_feature_level, x)
            
            # masked_X를 이번 step feature transformer에 input
            out = self.feat_transformers[step](masked_x)
            
            # feature transformer output에서 n_d(prediction layer 차원)만큼 가져온뒤 relu
            d = ReLU()(out[:, : self.n_d])
            
            # explain
            # 특징 중요도 계산 및 설명 업데이트:
            # 각 단계에서 예측 레이어의 중요도를 계산하고, 이를 M_explain에 더해 설명을 업데이트 
            
            # relu output matrix (n*D) -array 1*D로 만들기
            step_importance = torch.sum(d, dim=1)
            
            # mask * relu array
            M_explain += torch.mul(M_feature_level, step_importance.unsqueeze(dim=1))
            
            # update attention
            # 주의 정보 업데이트: 다음 단계의 주의 정보를 업데이트 
            att = out[:, self.n_d :]

        # 마스크 설명 및 주의 마스크 반환: 각 단계에서 얻은 마스크 설명(M_explain)과 주의 마스크들을 반환
        return M_explain, masks


# TabNet 네트워크의 디코더 부분을 정의하는 클래스 
class TabNetDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim,   # 입력 데이터의 특징 차원 
        n_d=8,   # 예측 레이어의 차원 
        n_steps=3,   # 디코더 단계 수 
        n_independent=1,   # 독립적인 GLU 레이어 수 
        n_shared=1,   # 공유되는 GLU 레이어 수 
        virtual_batch_size=128,   # GBN에서 사용할 배치 크기 
        momentum=0.02,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 1)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 1)
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        """
        
        # 모듈 초기화: torch.nn.Module의 초기화 함수를 호출하고, 클래스의 속성들을 정의
        # feat_transformers: 특징 변환기를 저장할 리스트를 생성 
        super(TabNetDecoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        self.feat_transformers = torch.nn.ModuleList()

        # 공유 특징 변환기: 공유할 GLU 레이어가 존재하는 경우, 이를 정의
        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                shared_feat_transform.append(Linear(n_d, 2 * n_d, bias=False))
                
        # 공유 변환기 없음: 공유 GLU 레이어가 없는 경우 None으로 설정 
        else:
            shared_feat_transform = None

        # 특징 변환기 생성: 각 단께마다 특징 변환기를 생성하여 리스트에 추가 
        for step in range(n_steps):
            transformer = FeatTransformer(
                n_d,
                n_d,
                shared_feat_transform,
                n_glu_independent=self.n_independent,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.feat_transformers.append(transformer)

        # 재구성 레이어: n_d 크기의 특징을 input_dim 크기로 변환하는 재구성 레이어를 정의 
        # 초기화: initialize_non_glu 함수를 사용해 가중치를 초기화 
        self.reconstruction_layer = Linear(n_d, self.input_dim, bias=False)
        initialize_non_glu(self.reconstruction_layer, n_d, self.input_dim)

    # 출력 초기화 및 단계별 변환: steps_output을 반복하여 각 단계의 출력을 변환하고, res에 더함
    def forward(self, steps_output):
        res = 0
        for step_nb, step_output in enumerate(steps_output):
            x = self.feat_transformers[step_nb](step_output)
            res = torch.add(res, x)
            
        # 최종 재구성: reconstruction_layer를 사용해 최종 결과를 재구성하고, 이를 반환 
        res = self.reconstruction_layer(res)
        return res


# TabNetPretraining: TabNet 모델의 사전 학습(pretraining)용 클래스 
class TabNetPretraining(torch.nn.Module):
    
    # 생성자: TabNetPretraining 객체의 초기화 함수로 여러 파라미터를 입력받음 
    def __init__(
        self,
        input_dim,   # 입력 데이터의 차원 
        pretraining_ratio=0.2,   # 마스킹 비율
        n_d=8,   # 예측 레이어의 차원
        n_a=8,   # 주의 레이어의 차원
        n_steps=3,   # 네트워크의 단계수 
        gamma=1.3,   # 주의도 갱신 스케일링 파라미터 
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        n_shared_decoder=1,
        n_indep_decoder=1,
        group_attention_matrix=None,
    ):
        
        # 상위 클래스 초기화: torch.nn.Module의 초기화를 호출 
        super(TabNetPretraining, self).__init__()

        # 카테고리 인덱스 및 차원 설정: 카테고리 특징 인덱스 및 차원과 임베딩 차원을 저장 
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        # 입력 파라미터 초기화: 생성자에서 받은 입력 파라미터들을 클래스 속성으로 저장
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type
        self.pretraining_ratio = pretraining_ratio
        self.n_shared_decoder = n_shared_decoder
        self.n_indep_decoder = n_indep_decoder

        # 유효성 검사: 단계 수와 독립 및 공유 레이어의 값이 적절한지 확인 
        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        # 임베딩 생성기: EnbeddingGenerator를 사용해 임베딩을 생성하고, 임베딩 후 차원을 저장 
        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        # 마스킹 오브젝트: 입력 데이터를 무작위로 마스킹하는 RandomObfuscator 객체를 초기화 
        self.masker = RandomObfuscator(self.pretraining_ratio,
                                       group_matrix=self.embedder.embedding_group_matrix)
        
        # 인코더: TabNetEncoder 객체를 초기화함 여러 파라미터를 전달하여 설정 
        self.encoder = TabNetEncoder(
            input_dim=self.post_embed_dim,
            output_dim=self.post_embed_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=self.embedder.embedding_group_matrix,
        )
        
        # 디코더: TabNetDecoder 객체를 초기화함 
        self.decoder = TabNetDecoder(
            self.post_embed_dim,
            n_d=n_d,
            n_steps=n_steps,
            n_independent=self.n_indep_decoder,
            n_shared=self.n_shared_decoder,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
        )

    # forward: 입력 데이터를 처리하는 메인 함수 
    def forward(self, x):
        """
        Returns: res, embedded_x, obf_vars
            res : output of reconstruction
            embedded_x : embedded input
            obf_vars : which variable where obfuscated
        """
        # 입력 임베딩: 입력 데이터를 임베딩 
        embedded_x = self.embedder(x)
        
        # 학습중 일때: 임베딩된 데이터를 마스킹하고, 마스킹된 데이터로 인코더-디코더를 실행후 결과와 임베딩,마스킹된 변수들을 반환 
        if self.training:
            masked_x, obfuscated_groups, obfuscated_vars = self.masker(embedded_x)
            # set prior of encoder with obfuscated groups
            prior = 1 - obfuscated_groups
            steps_out, _ = self.encoder(masked_x, prior=prior)
            res = self.decoder(steps_out)
            return res, embedded_x, obfuscated_vars
        
        # 학습중이 아닐 때: 마스킹 없이 인코더-디코더를 실행하고 결과와 임베딩된 데이터를 반환 
        else:
            steps_out, _ = self.encoder(embedded_x)
            res = self.decoder(steps_out)
            return res, embedded_x, torch.ones(embedded_x.shape).to(x.device)

    # forward_masks: 주의 마스크를 반환하는 함수, 임베딩된 데이터를 인코더의 forward_masks에 전달 
    def forward_masks(self, x):
        embedded_x = self.embedder(x)
        return self.encoder.forward_masks(embedded_x)


# TabNetEmbedding: 임베딩 없이 TabNet 네트워크의 주요 부분을 정의한 클래스 
class TabNetNoEmbeddings(torch.nn.Module):
    
    # 생성자: 네트워크의 여러 구성 요소와 파라미터들을 초기화 
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        input_dim : int
            Number of features
        output_dim : int or list of int for multi task classification
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        
        # 속성 초기화: 입력 및 출력 차원, 여러 하이퍼파라미터들을 클래스 속성으로 저장하고, 배치 정규화 레이어를 정의 
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_multi_task = isinstance(output_dim, list)
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type
        self.initial_bn = BatchNorm1d(self.input_dim, momentum=0.01)

        # 인코더: 입력 차원과 출력 차원, 여러 파라미터를 통해 TabNetEncoder를 초기화 
        self.encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
            group_attention_matrix=group_attention_matrix
        )

        # 멀티 태스크 매핑: 출력이 다중 작업인 경우, 각각의 작업에 대한 매핑 레이어를 정의 
        if self.is_multi_task:
            self.multi_task_mappings = torch.nn.ModuleList()
            for task_dim in output_dim:
                task_mapping = Linear(n_d, task_dim, bias=False)
                initialize_non_glu(task_mapping, n_d, task_dim)
                self.multi_task_mappings.append(task_mapping)
                
        # 최종 매핑: 단일 작업의 경우, 최종 매핑 레이어를 정의         
        else:
            self.final_mapping = Linear(n_d, output_dim, bias=False)
            initialize_non_glu(self.final_mapping, n_d, output_dim)

    # forward: 입력 데이터를 인코더에 전달하여 결과를 얻고, 여러 단계에서 나온 출력을 합산 
    def forward(self, x):
        res = 0
        steps_output, M_loss = self.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        # 멀티 태스크 출력: 다중 작업인 경우, 각 작업에 대해 별도의 출력값을 생성 
        if self.is_multi_task:
            # Result will be in list format
            out = []
            for task_mapping in self.multi_task_mappings:
                out.append(task_mapping(res))
                
        # 단일 출력: 단일 작업의 경우, 최종 매핑 레이어를 통해 결과를 반환 
        else:
            out = self.final_mapping(res)
        return out, M_loss

    # 주의 마스크를 반환하는 함수 
    def forward_masks(self, x):
        return self.encoder.forward_masks(x)


# TabNet 네트워크를 정의하는 클래스 
class TabNet(torch.nn.Module):
    
    def __init__(
        self,
        input_dim,   # 입력 데이터의 차원 
        output_dim,   # 네트워크의 출력 차원 
        n_d=8,   # 예측 레이어의 차원 
        n_a=8,   # 주의 레이어의 차원 
        n_steps=3,   # 네트워크 단계 수
        gamma=1.3,   # 주의 업데이트 스케일링 값
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=[],
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        group_attention_matrix : torch matrix
            Matrix of size (n_groups, input_dim), m_ij = importance within group i of feature j
        """
        
        # 상위 클래스 초기화: torch.nn.Module의 생성자를 호출하여 상위 클래스를 초기화 
        super(TabNet, self).__init__()
        
        # 카테고리 변수 초기화: 카테고리 인덱스와 차원, 임베딩 차원을 저장 
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        # 입력 및 출력, 하이퍼파라미터 초기화: 각 파라미터를 클래스 속성으로 저장 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        # 유효성 검사: n_steps가 0보다 커야 하고, n_independent 및 n_shared가 모두 0일 수 없음을 확인 
        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        # 임베더 생성: EmbeddingGenerator를 통해 카테고리 임베딩을 설정하고, 임베딩 후의 차원을 저장
        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim,
                                           cat_dims,
                                           cat_idxs,
                                           cat_emb_dim,
                                           group_attention_matrix)
        self.post_embed_dim = self.embedder.post_embed_dim

        # TabNet 인스턴스: 임베딩된 데이터를 사용하는 TabNetNoEmbeddings 객체를 초기화 
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
            self.embedder.embedding_group_matrix
        )

    # forward: 입력 데이터를 임베딩한 후, TabNetNoEmbeddings에서 처리
    def forward(self, x):
        x = self.embedder(x)
        return self.tabnet(x)

    # forward_masks: 입력 데이터를 임베딩한 후 마스크 출력을 반환
    def forward_masks(self, x):
        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


# 주의 메커니즘을 적용하는 변환기 클래스 
class AttentiveTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        group_dim,
        group_matrix,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        group_dim : int
            Number of groups for features
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        
        # 상위 클래스 초기화: troch.nn.Module의 생성자를 호출 
        super(AttentiveTransformer, self).__init__()
        
        # 완전 연결 레이어: 입력과 그룹 차원에 맞춘 완전 연결(FC) 레이어를 초기화
        self.fc = Linear(input_dim, group_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, group_dim)
        
        # 배치 정규화: 그룹 차원에 맞춘 고스트 배치 정규화(GBN) 레이어를 정의 
        self.bn = GBN(
            group_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

        # 마스킹 함수 선택: sparsemax 또는 entmax 마스킹 함수를 선택 
        if mask_type == "sparsemax":
            # Sparsemax
            self.selector = sparsemax.Sparsemax(dim=-1)
        elif mask_type == "entmax":
            # Entmax
            self.selector = sparsemax.Entmax15(dim=-1)
        else:
            raise NotImplementedError(
                "Please choose either sparsemax" + "or entmax as masktype"
            )

    # forward: 입력 특징을 완전 연결(FC) 레이어와 배치 정규화를 거쳐, priors와 곱한 후 마스킹함
    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.selector(x)
        return x


# 특징 변환을 담당하는 클래스 
class FeatTransformer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        shared_layers,
        n_glu_independent,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        
        # 상위 클래스 초기화: torch.nn.Module의 생성자를 호출 
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        input_dim : int
            Input size
        output_dim : int
            Output_size
        shared_layers : torch.nn.ModuleList
            The shared block that should be common to every step
        n_glu_independent : int
            Number of independent GLU layers
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization within GLU block(s)
        momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        # 파라미터 설정: 독립적인 GLU 레이어수, 배치 크기, 모멘텀 등을 설정 
        params = {
            "n_glu": n_glu_independent,
            "virtual_batch_size": virtual_batch_size,
            "momentum": momentum,
        }

        # 공유 레이어 설정: 공유 레이어가 주어지지 않은 경우 Identity 레이어를 사용
        # 그렇지 않으면 GLU_Block을 사용 
        if shared_layers is None:
            # no shared layers
            self.shared = torch.nn.Identity()
            is_first = True
        else:
            self.shared = GLU_Block(
                input_dim,
                output_dim,
                first=True,
                shared_layers=shared_layers,
                n_glu=len(shared_layers),
                virtual_batch_size=virtual_batch_size,
                momentum=momentum,
            )
            is_first = False

        if n_glu_independent == 0:
            # no independent layers
            self.specifics = torch.nn.Identity()
        else:
            spec_input_dim = input_dim if is_first else output_dim
            self.specifics = GLU_Block(
                spec_input_dim, output_dim, first=is_first, **params
            )

    # forward: 입력 데이터를 공유 레이어와 독립적인 GLU 레이어를 거쳐 처리한 후 결과를 반환
    def forward(self, x):
        x = self.shared(x)
        x = self.specifics(x)
        return x


# 이 클래스는 각 단계에 대해 독립적으로 작동하는 GLU블록 정의 
class GLU_Block(torch.nn.Module):
    """
    Independent GLU block, specific to each step
    """

    # GLU 블록의 초기화 함수로, 여러 파라미터 설정 
    def __init__(
        self,
        input_dim,
        output_dim,
        n_glu=2,
        first=False,
        shared_layers=None,
        virtual_batch_size=128,
        momentum=0.02,
    ):
        super(GLU_Block, self).__init__()
        
        # first,shared_layers,n_glu 등의 인자를 저장하고 GLU 레이어들을 저장할 ModuleList 객체를 생성 
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        # 가상 배치 크기와 모멘텀 등의 파라미터를 사전으로 정의 
        params = {"virtual_batch_size": virtual_batch_size, "momentum": momentum}

        # 공유 레이어가 있으면 첫 번째 공유 레이어를 사용하여 첫 GLU 레이어를 추가하고, 없으면 None으로 설정 
        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim, fc=fc, **params))
        
        # GLU 레이어의 수에 따라 추가 GLU 레이블들을 반복해서 설정 
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim, fc=fc, **params))

    # forward 메서드는 블록이 데이터를 처리하는 과정을 정의하며, 연산의 스케일을 줄이기 위해 0.5의 제곱근을 계산 
    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        
        # 블록이 첫 번째 레이어라면 첫 GLU 레이어만 적용하고 나머지 레이어들을 처리할 준비를 함 
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        # 각 GLU 레이어의 출력을 추가하고 스케일 조정을 적용 
        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x * scale
        return x


# GLU_Layer는 입력과 출력 차원을 정의하며, FC 레이어를 설정
# 기본 배치 크기와 모멘텀도 설정됨 
class GLU_Layer(torch.nn.Module):
    def __init__(
        self, input_dim, output_dim, fc=None, virtual_batch_size=128, momentum=0.02
    ):
        
        # FC 레이어가 주어지면 그대로 사용하고, 없으면 새로 생성함. GLU를 초기화함 
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2 * output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2 * output_dim)

        # 배치 정규화 층을 추가 
        self.bn = GBN(
            2 * output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum
        )

    # forward 메서드는 FC 레이어와 배치 정규화를 적용한 후, 시그모이드 함수를 사용해 GLU 연산을 수행 
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, : self.output_dim], torch.sigmoid(x[:, self.output_dim :]))
        return out


# 이 클래스는 입력 데이터에 대한 임베딩을 생성하는 기능 
class EmbeddingGenerator(torch.nn.Module):
    """
    Classical embeddings generator
    """

    # EmbeddingGenerator의 초기화로, 입력 데이터와 범주형 데이터에 대한 정보를 저장하고 그룹 행렬을 사용
    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dims, group_matrix):
        """This is an embedding module for an entire set of features

        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : list of int
            Embedding dimension for each categorical features
            If int, the same embedding dimension will be used for all categorical features
        group_matrix : torch matrix
            Original group matrix before embeddings
        """
        super(EmbeddingGenerator, self).__init__()

        # 범주형 데이터가 없으면 임베딩을 건너뜀. 임베딩을 하지 않는 경우에는 그대로 입력 차원을 유지 
        if cat_dims == [] and cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            self.embedding_group_matrix = group_matrix.to(group_matrix.device)
            return
        else:
            self.skip_embedding = False

        # 임베딩을 적용할 경우, 최종 임베딩 차원을 계산하고 범주형 특성에 대해 임베딩을 저장 
        self.post_embed_dim = int(input_dim + np.sum(cat_emb_dims) - len(cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # 각 범주형 특성에 대해 임베딩을 생성 
        for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))


        # record continuous indices
        # 연속형 특성과 범주형 특성을 구분하기 위한 마스크를 만듬 
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0


        # update group matrix
        # 그룹 행렬을 임베딩된 특성 차원에 맞게 변환함
        n_groups = group_matrix.shape[0]
        self.embedding_group_matrix = torch.empty((n_groups, self.post_embed_dim),
                                                  device=group_matrix.device)
        
        # 그룹 행렬의 크기를 재조정하고 임베딩된 특성에 맞춰 변환된 그룹 행렬을 계산
        for group_idx in range(n_groups):
            post_emb_idx = 0
            cat_feat_counter = 0
            for init_feat_idx in range(input_dim):
                if self.continuous_idx[init_feat_idx] == 1:
                    # this means that no embedding is applied to this column
                    self.embedding_group_matrix[group_idx, post_emb_idx] = group_matrix[group_idx, init_feat_idx]  # noqa
                    post_emb_idx += 1
                else:
                    # this is a categorical feature which creates multiple embeddings
                    n_embeddings = cat_emb_dims[cat_feat_counter]
                    self.embedding_group_matrix[group_idx, post_emb_idx:post_emb_idx+n_embeddings] = group_matrix[group_idx, init_feat_idx] / n_embeddings  # noqa
                    post_emb_idx += n_embeddings
                    cat_feat_counter += 1

    # 임베딩이 필요하지 않으면 입력 데이터를 그대로 반환 
    def forward(self, x):
        """
        Apply embeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x

        # 입력 데이터를 각각 연속형 또는 범주형 특성에 맞게 변환하여 임베딩을 적용한 후, 최종 임베딩을 반환
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(
                    self.embeddings[cat_feat_counter](x[:, feat_init_idx].long())
                )
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings


# 주어진 비율에 따라 그룹 수준에서 임의로 특성의 일부를 마스킹하는 역할을함
class RandomObfuscator(torch.nn.Module):
    """
    Create and applies obfuscation masks.
    The obfuscation is done at group level to match attention.
    """

    # 마스킹 비율과 그룹 행렬을 초기화하며, 그룹의 개수를 저장 
    def __init__(self, pretraining_ratio, group_matrix):
        """
        This create random obfuscation for self suppervised pretraining
        Parameters
        ----------
        pretraining_ratio : float
            Ratio of feature to randomly discard for reconstruction

        """
        super(RandomObfuscator, self).__init__()
        self.pretraining_ratio = pretraining_ratio
        # group matrix is set to boolean here to pass all posssible information
        self.group_matrix = (group_matrix > 0) + 0.
        self.num_groups = group_matrix.shape[0]

    # forward 메서드는 마스킹을 적용하며, 배치 크기를 저장 
    def forward(self, x):
        """
        Generate random obfuscation mask.

        Returns
        -------
        masked input and obfuscated variables.
        """
        bs = x.shape[0]

        # 주어진 마스킹 비율에 따라 그룹을 임의로 마스킹할 확률을 계산 
        obfuscated_groups = torch.bernoulli(
            self.pretraining_ratio * torch.ones((bs, self.num_groups), device=x.device)
        )
        
        # 마스킹된 그룹에 따라 입력 데이터를 마스킹하여 마스킹된 변수를 생성하고, 이를 반환
        obfuscated_vars = torch.matmul(obfuscated_groups, self.group_matrix)
        masked_input = torch.mul(1 - obfuscated_vars, x)
        
        return masked_input, obfuscated_groups, obfuscated_vars
