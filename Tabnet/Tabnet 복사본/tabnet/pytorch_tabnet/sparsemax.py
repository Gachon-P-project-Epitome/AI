from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

import torch

"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


# credits to Yandex https://github.com/Qwicen/node/blob/master/lib/nn_utils.py
# 이 함수는 입력 텐서 (input)의 크기에 맞는 인덱스 텐서를 생성
# dim=0 은 기본적으로 첫 번째 차원에서 인덱스를 생성하도록 지정된 매개변수 
def _make_ix_like(input, dim=0):
    
    # input.size(dim)을 통해, input텐서의 dim 차원의 크기를 d에 저장
    # 즉 dim 차원의 크기를 알아내는 것
    d = input.size(dim)
    
    # rho는 1부터 d까지의 값을 가지는 1D 텐서를 생성, 이 텐서는 input과 같은 장치(device)와 데이터 타입(dtype)을 사용
    # 이 텐서는 Sparsemax 에서 계산에 사용될 정렬된 인덱스 역할을함
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    
    # input 텐서와 같은 차원의 리스트 view를 생성하며, 모든 차원에 대해 값이 1로 초기화됨
    # 이 리스트는 rho 텐서를 올바른 차원으로 reshape 하기 위한 설정
    view = [1] * input.dim()
    
    # view 의 첫번째 차원(view[0])을 -1로 설정, 이는 rho 텐서를 그 차원에서 broadcast 할 수 있게 해줌
    # view 는 reshape할 텐서의 크기를 지정하는 역할
    view[0] = -1
    
    # rho.view(view)는 rho 텐서를 view에 맞게 reshape함 즉, rho는 이제 원하는 모양으로 재구성됨
    # transpose(0, dim)은 rho 텐서의 첫 번째 차원(0)과 dim 차원을 교환하여 결과 텐서를 반환
    # 이는 Sparsemax에서 정렬된 인덱스를 해당 차원에서 사용할 수 있도록 설정하는 과정
    return rho.view(view).transpose(0, dim)


class SparsemaxFunction(Function):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    # forward 메서드는 순전파를 정의
    # ctx는 PyTorch에서 자동 미분을 위해 사용되는 컨텍스트 객체로, 역전파를 위한 중간 결과를 저장하는데 사용
    # input은 함수의 입력된 텐서이고, dim=-1은 Sparsemax가 적용될 차원을 나타냄, 기본값으로 마지막 차원 설정 
    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        # 전달받은 dim 값을 ctx에 저장, 나중에 역전파에서 사용하기 위해 보관
        ctx.dim = dim
        
        # 입력텐서 input의 지정된 dim에서 최댓값(max_val)을 구함
        # keepdim=True는 해당 차원을 유지한 채로 최댓값을 반환하도록 설정
        max_val, _ = input.max(dim=dim, keepdim=True)
        
        # 계산중 오버플로우를 방지하기 위해 입력 텐서에서 최댓값을 뺌 
        input -= max_val  # same numerical stability trick as for softmax
        
        # _threshold_and_support를 호출하여 tau(임계값)와 support size (지원크기) 를 계산 이 값들은 sparsemax 변환에서 중요
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        
        # input에서 계산된 tau를 뺀 후, 결과를 0 이상의 값으로 제한
        # 이 작업을 통해 음수값을 0으로 만들며 희소성을 구현하는데 중요
        output = torch.clamp(input - tau, min=0)
        
        # 역전파에서 사용할 값을 ctx에 저장
        ctx.save_for_backward(supp_size, output)
        
        # 정규화된 출력 텐서 반환
        return output

    # backward 함수는 자동 미분 과정에서 역전파 계산을 정의 
    # ctx:순전파(forward)에서 저장된 컨텍스트 객체로, 역전파에서 필요한 데이터를 저장하고 불러오는데 사용
    # grad_output:출력에 대한 손실의 그레이디언트가 입력으로 주어짐, 이는 역전파 과정에서 전달
    @staticmethod
    def backward(ctx, grad_output):
        
        # 위의 순전파 과정에서 ctz.save_for_backward()로 저장된 supp_size,output을 불러옴
        # supp_size : sparsemax의 서포트 크기(희소성을 가진 값의 개수)
        # output : 순전파 결과 텐서 
        supp_size, output = ctx.saved_tensors
        
        # 순전파에서 저장한 dim값을 불러옴(sparsemax가 적용된 차원)
        dim = ctx.dim
        
        # grad_input의 복사본(clone)을 만듬 
        # (역전파 과정에서 값을 수정해야 하기 때문에 원본을 유지하기 위해 복사본 사용)
        grad_input = grad_output.clone()
        
        # 순전파에서 희소화된값(output==0)에 해당하는 그레이디언트 값을 0으로 만듬
        # sparsemax의 특성상 0이 된 값들에 대한 그레이디언트는 계산에 영향을 미치지 않기 때문에 해당 값을 0으로 설정해 무시 
        grad_input[output == 0] = 0

        # grad_input에서 희소하지 않은 값들에 대한 평균 그레이디언트 v_hat을 계산
        # grad_input.sum(dim=dim) : 저장된 차원 dim에서 그레이디언트를 모두 더함
        # supp_size.to(output.dtype) : supp_size를 output의 데이터 타입으로 변환
        # .squeeze() : 불필요한 차원을 제거해 값을 간결하게 만듬
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        
        #v_hat을 지정된 dim에 맞게 차원을 확장, 이렇게 하면 v_hat이 grad_input과 같은 차원을 가짐
        v_hat = v_hat.unsqueeze(dim)
        
        # output != 0 인 경우에는 grad_input - v_hat
        # output == 0 인 경우에는 grad_input을 그대로 유지 
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        
        # 최종적으로 계산된 입력에 대한 그레이디언트 grad_input을 반환 
        # None은 dim값이 상수이기 때문에 그레이디언트가 필요없음
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor    입력텐서로 모든 차원을 가질 수 있음 
            any dimension
        dim : int              sparsemax를 적용할 차원
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor           임계값으로, 희소화 기준이 되는 값
            the threshold value
        support_size : torch.Tensor  서포트 크기로, 희소화된 값의 개수를 나타냄

        """
        
        # input 텐서를 지정된 dim 차원에서 내림차순으로 정렬
        # torch.sort()는 정렬된 텐서(input_srt)와 그에 따른 인덱스를 반환하는데 여기서는 인덱스는 사용하지 않기 때문에 _로 무시
        # 즉, input_srt는 입력 텐서를 큰 값부터 작은 값 순서로 정렬한 텐서
        input_srt, _ = torch.sort(input, descending=True, dim=dim)
        
        # input_srt의 dim 차원에서 누적 합(cumulative sum)을 계산한뒤, 그 값에서 1을 뺌
        # cumsum(dim)은 해당 차원의 원소들을 순서대로 더해 누적된 값을 계산
        # input_cumsum은 imput_srt의 누적 합에서 1을 뺀 텐서, 이는 이후 임계값 계산에 사용
        input_cumsum = input_srt.cumsum(dim) - 1
        
        # input 텐서와 같은 크기의 인덱스 텐서를 만듬
        # _make_ix_like() 함수는 각 언소의 인덱스를 해당 차원 dim에 맞춰 생성하는 것
        # 예를 들어, input이 [5,4,3] 일때, rhos는 [1,2,3]과 같은 인덱스를 가지는 텐서일 가능성
        rhos = _make_ix_like(input, dim)
        
        # rhos 텐서와 input_srt를 곱한 결과가 input_cumsum보다 큰지 여부를 판단하는 불리언 마스크 생성
        # 즉, 각 요소가 해당 누적 합을 초과하는지 확인하고, 초과하면 True, 그렇지 않으면 False를 반환하는 텐서를 만듬
        # 이 텐서는 sparsemax 연산에서 유효한 값들이 어느 위치에 있는지를 나타내는 역할을 함
        support = rhos * input_srt > input_cumsum
        
        # support 텐서에서 True의 개수를 해당 차원 dim에서 계산
        # sum(dim=dim)은 해당 차원에서 True의 개수를 더해 그 크기를 구하는 것
        # unsqueeze(dim)은 이 차원에 새로운 차원을 추가하여 텐서의 차원을 늘림
        # support_size는 sparsemax연산에서 유효한 값들의 개수를 나타냄
        support_size = support.sum(dim=dim).unsqueeze(dim)
        
        # gather()함수는 support_size - 1 에 해당하는 위치의 값을 input_cumsum에서 가져옴
        # 즉, 각 위치에서 서포트 크기만큼의 누적 합 중 마지막 값을 tau로 설정, 이것이 sparsemax에서의 임게값
        tau = input_cumsum.gather(dim, support_size - 1)
        
        # 임계값 tau를 서포트 크기 support_size로 나눔
        # support_size의 데이터 타입을 input의 데이터 타입으로 변환한 후 나누기를 수행
        # 이는 sparsemax에서 최종적으로 각 요소를 임계값으로 나누어 희소화된 값을 구하는 과정
        tau /= support_size.to(input.dtype)
        
        # 최종적으로 계산된 임계값 tau와 서포트 크기 support_size를 반환
        return tau, support_size

# SparsemaxFunction 클래스의 apply 메서드를 sparsemax라는 이름으로 할당
# 이는 SparsemaxFunction 이라는 커스텀 함수의 실제 동작을 수행하는 부분으로 forward 및 backward 연산 관리
# 이렇게 할당하면 이후 코드에서 sparsemax(input, dim)형태로 간단하게 호출할 수 있음
sparsemax = SparsemaxFunction.apply

# PyTorch의 nn.Module을 상속하는 Sparsemax 클래스를 정의
# nn.Module은 PyTorch에서 신경망 모델의 기본 빌딩 블록 역할을 하며, 모든 사용자 정의레이어는 이 클래스를 상속받아야 함
class Sparsemax(nn.Module):

    # Sparsemax 클래스의 생성자 메서드로, 인스턴스를 초기화 
    # dim은 sparsemax를 적용할 차원을 나타내며, 기본값은 -1로 마지막 차원을 의미
    def __init__(self, dim=-1):
        
        # 입력으로 받은 dim 값을 클래스 변수 self.dim에 저장
        # 이후 forward 메서드에서 이 값을 사용하여 sparsemax 연산이 적용될 차원을 지정
        self.dim = dim
        
        # 부모 클래스인 nn.Module의 생성자 (__init__())를 호출
        # 이를 통해 Sparsemax 클래스가 nn.Module의 기능을 제대로 상속받아 사용할 수 있게 됨
        # 특히, 파라미터 등록이나 모듈 관리 등 nn.Module에서 제공하는 중요한 기능들을 사용할 수 있음
        super(Sparsemax, self).__init__()
    
    # PyTorch의 모든 nn.Module 클래스는 forward 메서드를 통해 입력을 처리
    # 이 메서드는 모델이 학습 중일 때 순전파(forward pass)시 호출
    # 여기서 input은 신경망의 입력 값
    def forward(self, input):
        
        # 입력 input에 대해 sparsemax 연산을 수행하고 그 결과를 반환
        # 앞서 정의한 sparsemax = SparsemaxFunction.apply를 통해 sparsemax 계산이 적용
        # 이때 self.dim 값을 사용하여 특정 차원에 대해 연산을 수행 
        return sparsemax(input, self.dim)

# Entmax15Function 클래스는 PyTorch의 Function 클래스를 상속받아 Entmax(alpha=1.5) 구현
class Entmax15Function(Function):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    # forward 메서드는 순전파를 정의, 여기서는 ctx는 컨텍스트 객체로, 역전파 시 필요한 값들을 저장
    # input은 입력텐서, dim은 Entmax가 적용될 차원을 지정하며 기본값은 -1로 마지막 차원을 의미
    # ctx.dim에 dim값을 저장해 나중에 backward에서 사용할 수 있게 함
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        
        # 입력 텐서에서 해당 차원의 최대값을 찾음, 이는 나중에 수치적 안정성을 위해 사용
        # keepdim=True는 차원을 유지한 채 최대값을 반환하게 함
        max_val, _ = input.max(dim=dim, keepdim=True)
        
        # 입력텐서에서 최대값을 빼줌, 이는 소프트맥스 계산에서 흔히 사용되는 수치적 안정성 트릭으로 오버플로우 방지 
        input = input - max_val  # same numerical stability trick as for softmax
        
        # Entmax 연산에 맞게 입력을 2로 나눔, 이는 Entmax의 alpha 값이 1.5일때 수식적으로 필요한 변환
        input = input / 2  # divide by 2 to solve actual Entmax

        # _threshold_and_support 함수는 tau_star와 support_size를 계산
        # 이는 Sparsemax 처럼 특정 임계값(tau_star)을 구해 희소화를 적용하는데 사용
        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        
        # 입력에서 tau_star를 빼고, 그 값을 0으로 최소화한 뒤 제곱
        # 이 과정을 통해 음수는 모두 0으로 처리되고, 희소화된 값들이 제곱
        output = torch.clamp(input - tau_star, min=0) ** 2
        
        # output을 컨텍스트에 저장하여 나중에 역전파(backward) 단계에서 사용할 수 있도록함
        ctx.save_for_backward(output)
        
        # 최종적으로 계산된 값을 반환
        return output

    # backward 메서드는 역전파를 정의, 여기서 grad_output은 순전파에서의 출력에 대한 손실 함수의 기울기
    # ctx.saved_tensors에서 저장된 Y 값을 불러옴, 이 Y는 순전파에서 계산된 ouput 값
    @staticmethod
    def backward(ctx, grad_output):
        Y, = ctx.saved_tensors
        
        # Y의 제곱근을 구하여 gppr를 계산, 이는 이차 도함수(g''(Y))의 역수와 같은 역할
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        
        #입력에 대한 기울기(dX)를 grad_output과 gqqr의 곱으로 계산
        dX = grad_output * gppr
        
        # dX와 gppr의 합을 구해 q 값을 계산, 이 값은 이후 기울기 보정에 사용
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        
        # q의 차원을 확장하여 ctx.dim 위치에 새로운 차원을 추가
        # 이렇게 하면 차원이 맞아 기울기 계산 시 문제가 발생하지 않음
        q = q.unsqueeze(ctx.dim)
        
        # 기울기 보정을 위해 dX에서 q * gppr 값을 빼줌 이는 Entmax에서의 역전파를 정확하게 수행하기 위한 계산
        dX -= q * gppr
        
        # dX(입력에 대한 기울기)를 반환하고, 두번째 값은 None
        # 두번째 반환 값은 dim에 대한 기울기인데, dim은 학습 가능한 값이 아니기 때문에 None으로 반환
        return dX, None
    
    # 입력 텐서를 내림차순으로 정렬, Xsrt는 정렬된 텐서 
    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = torch.sort(input, descending=True, dim=dim)

        # input과 같은 크기의 인덱스 텐서를 만듬 
        # 이는 각 차원의 인덱스를 rho라는 텐서로 저장
        rho = _make_ix_like(input, dim)
        
        # 정렬된 텐서 Xsrt에 대해 누적합을 구하고, 이를 rho로 나눠서 평균을 계산
        mean = Xsrt.cumsum(dim) / rho
        
        # 정렬된 텐서 Xsrt의 제곱에 대해 누적 합을 구하고, 이를 rho로 나눠서 제곱된 평균을 계산
        mean_sq = (Xsrt ** 2).cumsum(dim) / rho
        
        # mean_sq와 mean의 차이를 이용해 ss(분산)을 계산
        ss = rho * (mean_sq - mean ** 2)
        
        # 분산에 기반해 delta 값을 계산, 이는 희소화에서 임계값을 계산하는데 사용
        delta = (1 - ss) / rho

        # NOTE this is not exactly the same as in reference algo
        # Fortunately it seems the clamped values never wrongly
        # get selected by tau <= sorted_z. Prove this!
        
        # delta 값을 0이상으로 클램핑하여 음수를 제거
        delta_nz = torch.clamp(delta, 0)
        
        # tau 값을 계산하는데, 이는 mean에서 delta_nz의 제곱근을 뺀 값
        tau = mean - torch.sqrt(delta_nz)

        # tau 값이 Xsrt보다 작은 위치의 개수를 구하고, 이를 서포트 크기로 설정
        support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
        
        # support_size에 기반해 tau_star 값을 선택
        tau_star = tau.gather(dim, support_size - 1)
        
        # 최종적으로 계산된 tau_star와 support_size를 반환
        return tau_star, support_size

# Entmoid15는 PyTorch의 Function 클래스를 상속받아 구현된 커스텀 함수
# 주석에서 설명하듯이 lambda x: Entmax15([x, 0])와 같은 역할을 하지만 최적화된 구현
# 즉, 입력값 x와 0사이에서 Entmax15를 계산하는 최적화된 함수 
class Entmoid15(Function):
    """ A highly optimized equivalent of lambda x: Entmax15([x, 0]) """

    # forward 메서드는 순전파를 정의
    # input 텐서를 Entmoid15._forward 메서드로 넘겨 계산한 결과를 output에 저장
    # ctx.save_for_backward(output)를 통해 역전파에서 사용할 output 값을 저장
    # 최종적으로 계산된 output을 반환
    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    # input의 절대값을 계산하여 부호에 관계없이 다룸
    # is_pos는 input이 양수인지 여부를 나타내는 불리언 텐서
    # input >= 0 이면 True, 그렇지 않으면 False
    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        
        # tau는 input과 8 - input ** 2 의 제곱근을 더한 값을 2로 나눈것
        # F.relu()는 8 - input ** 2 에서 음수를 제거하고, 0 이상만 남김 
        # 이 과정은 수치적 안정성을 위해 사용됨
        tau = (input + torch.sqrt(F.relu(8 - input ** 2))) / 2
        
        # tau가 input보다 작거나 같으면 그 위치에 2.0을 채움
        # tau 값을 input에 따라 수정하는 과정 
        tau.masked_fill_(tau <= input, 2.0)
        
        # tau - input 값에서 음수를 제거하고, 그 값을 제곱한 후 0.25를 곱한 결과를 y_neg에 저장
        # 이 계산은 Entmax의 비양수 부분(즉, 음수 부분)을 처리하는 방식
        y_neg = 0.25 * F.relu(tau - input, inplace=True) ** 2
        
        # is_pos값에 따라 결과를 반환
        # input이 양수(is_pos=True)면 1 - y_neg을 반환
        # input이 음수(is_pos=False)aus y_neg을 반환
        # 즉, 입력값이 양수면 1에서 비양수 부분을 뺀 값을 반환하고, 음수면 y_neg 값을 그대로 반환
        # 이는 함수 출력이 [0, 1] 범위 내에서 유지되도록 보장
        return torch.where(is_pos, 1 - y_neg, y_neg)

    # backward 메서드는 역전파를 정의
    # ctx.saved_tensors[0] 에서 순전파에서 저장된 output 값을 불러오고, 이를 Entmoid15._backward 메서드로 넘겨 기울기 계산
    # 이 메서드는 순전파에서의 출력과 손실 함수의 기울기인 grad_output을 사용하여 역전파 계산을 수행 
    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    # output의 제곱근과 1 - output의 제곱근을 각각 계산하여 gppr0와 gppr1에 저장
    # 이는 Entmax의 역전파 공식에 따른 계산
    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        
        # 입력에 대한 기울기(grad_input)는 순전파에서 출력된 기울기(grad_output)에 gppr0를 곱한 값
        grad_input = grad_output * gppr0
        
        # grad_input을 gppr0와 gppr1의 합을 나눈 값을 q에 저장, 이는 역전파 과정에서 필요한 중간 값
        q = grad_input / (gppr0 + gppr1)
        
        # grad_input에서 q * gppr0를 빼주어 기울기 보정을 수행
        # 이를 통해 입력에 대한 정확한 기울기를 계산 
        grad_input -= q * gppr0
        
        # 최종적으로 계산된 기울기 grad_input을 반환
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply

# Entmax15 클래스는 PyTorch의 nn.Module을 상속받아 정의된 모듈
# 이 클래스는 신경망에서 Entmax를 사용할 수 있도록 모듈 형태로 구현
class Entmax15(nn.Module):

    # 생성자 메서드로, dim이라는 인자를 받음 이 dim은 Entmax15가 적용될 차원을 나타내며 기본값은 -1로 마지막 차원을 의미
    # self.dim에 dim값을 저장하여 나중에 사용할 수 있도록함
    # super(Entmax15, self).__init__()를 호출하여 부모 클래스인 nn.Module의 생성자를 초기화 
    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    # forward 메서드는 순전파 정의
    # 입력값 input에 대해 entmax15 함수를 적용 즉, Entmax15Function.apply(input, self.dim)이 호출
    # input 텐서에 Entmax 연산이 적용되어 반환 
    def forward(self, input):
        return entmax15(input, self.dim)


# Credits were lost...
# def _make_ix_like(input, dim=0):
#     d = input.size(dim)
#     rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
#     view = [1] * input.dim()
#     view[0] = -1
#     return rho.view(view).transpose(0, dim)
#
#
# def _threshold_and_support(input, dim=0):
#     """Sparsemax building block: compute the threshold
#     Args:
#         input: any dimension
#         dim: dimension along which to apply the sparsemax
#     Returns:
#         the threshold value
#     """
#
#     input_srt, _ = torch.sort(input, descending=True, dim=dim)
#     input_cumsum = input_srt.cumsum(dim) - 1
#     rhos = _make_ix_like(input, dim)
#     support = rhos * input_srt > input_cumsum
#
#     support_size = support.sum(dim=dim).unsqueeze(dim)
#     tau = input_cumsum.gather(dim, support_size - 1)
#     tau /= support_size.to(input.dtype)
#     return tau, support_size
#
#
# class SparsemaxFunction(Function):
#
#     @staticmethod
#     def forward(ctx, input, dim=0):
#         """sparsemax: normalizing sparse transform (a la softmax)
#         Parameters:
#             input (Tensor): any shape
#             dim: dimension along which to apply sparsemax
#         Returns:
#             output (Tensor): same shape as input
#         """
#         ctx.dim = dim
#         max_val, _ = input.max(dim=dim, keepdim=True)
#         input -= max_val  # same numerical stability trick as for softmax
#         tau, supp_size = _threshold_and_support(input, dim=dim)
#         output = torch.clamp(input - tau, min=0)
#         ctx.save_for_backward(supp_size, output)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         supp_size, output = ctx.saved_tensors
#         dim = ctx.dim
#         grad_input = grad_output.clone()
#         grad_input[output == 0] = 0
#
#         v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
#         v_hat = v_hat.unsqueeze(dim)
#         grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
#         return grad_input, None
#
#
# sparsemax = SparsemaxFunction.apply
#
#
# class Sparsemax(nn.Module):
#
#     def __init__(self, dim=0):
#         self.dim = dim
#         super(Sparsemax, self).__init__()
#
#     def forward(self, input):
#         return sparsemax(input, self.dim)
