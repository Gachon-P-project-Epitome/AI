# Author: Arnaud Joly, Joel Nothman, Hamzeh Alsalhi
#
# License: BSD 3 clause
"""
Multi-class / multi-label utility function
==========================================

"""
from collections.abc import Sequence
from itertools import chain

from scipy.sparse import issparse
from scipy.sparse.base import spmatrix
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np
import pandas as pd


# X는 검사할 입력 배열이고, allow_nan은 NaN을 허용할지 여부를 결정
def _assert_all_finite(X, allow_nan=False):
    """Like assert_all_finite, but only for ndarray."""

    # 입력데이터를 NumPy 배열로 변환
    # 이미 배열인 경우에는 원래 배열을 유지하고, 배열이 아닌 경우는 배열로 변환
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method. The sum is also calculated
    # safely to reduce dtype induced overflows.
    # 오버플로우를 방지하기 위해 필요 시 np.isfinite로 대체
    
    # 배열 X의 데이터 타입이 float 또는 복소수 타입인지 확인 (FMA는 복소수)
    # dtype.kind는 데이터 유형을 나타내는 코드이고, "f"는 부동소수점, "c" 는 복소수 의미
    is_float = X.dtype.kind in "fc"
    
    # X가 부동소수점 또는 복소수 배열이면서 모든 요소의 합이 유한한지 확인
    # 합이 유한하면 추가 작업 없이 넘어감 (pass는 아무 동작도 하지 않음을 의미)
    if is_float and (np.isfinite(np.sum(X))):
        pass
    
    # 배열이 부동소수점 또는 복소수인데, 앞서 합이 유한하지 않으면 에러 메시지 형식을 미리 정의
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        
        # 여기서 allow_nan이 True일 때 배열에 무한대(inf)가 있거나
        # allow_nan이 False일 때 배열에 유한하지 않은 값(NaN, inf)이 있으면 조건이 참이 됨
        if (
            allow_nan
            and np.isinf(X).any()
            or not allow_nan
            and not np.isfinite(X).all()
        ):
            
            # NaN을 허용하는 경우 오류 유형을 "infinity"로 설정하고 그렇지 않은 경우 "NaN, infinity"로 설정
            type_err = "infinity" if allow_nan else "NaN, infinity"
            
            # 유효하지 않은 값이 발견되면 에러 메시지 발생 
            # 메시지에는 문제가 되는 값 유형과 배열의 데이터 타입이 포함
            raise ValueError(msg_err.format(type_err, X.dtype))
        
    # for object dtype data, we only check for NaNs (GH-13254)
    # X가 object 타입 배열이고 NaN을 허용하지 않을 때, 배열에 NaN이 있으면 에러 메시지 발생
    elif X.dtype == np.dtype("object") and not allow_nan:
        if np.isnan(X).any():
            raise ValueError("Input contains NaN")


# X에 NaN이나 무한대(inf)가 있으면 ValueError를 발생시킴
# X: 배열 또는 희소 행렬(sparse matrix)
# allow_nan: True일 경우 NaN을 허용하고, False일 경우 NaN을 허용하지 않음
def assert_all_finite(X, allow_nan=False):
    """Throw a ValueError if X contains NaN or infinity.

    Parameters
    ----------
    X : array or sparse matrix
    allow_nan : bool
    """
    
    # X가 희소 행렬(sparse matrix)인지 확인하고 그렇다면 그 데이터(X.data)만 전달하고, 아니면 그대로 X를 전달
    _assert_all_finite(X.data if sp.issparse(X) else X, allow_nan)


# 이 함수는 다중 클래스(multiclass) 데이터를 처리할 때 고유한 클래스 라벨을 반환하는 역할을함
# y가 Numpy 배열 속성을 가지고 있으면 (__array__ 속성), np.unique()를 사용하여 고유한 라벨을 반환
# y가 배열이 아닌 경우에는 집합(set)으로 변환하여 고유한 라벨을 반환 
def _unique_multiclass(y):
    if hasattr(y, "__array__"):
        return np.unique(np.asarray(y))
    else:
        return set(y)


# 멀티라벨 지시자(multilabel indicator) 관련 처리를 위해 작성됬지만 아직 구현이 안됨
# 호출될 때 IndexError를 발생시키며, 에러 메시지에는 입력 라벨 y의 크기가 (n_samples,)이어야 한다는 내용과 함께
# 멀티라벨 분류를 시도 중이라면 TabNetMultiTaskClassification 또는 TabNetRefressor를 사용하라는 제안을 포함
def _unique_indicator(y):
    """
    Not implemented
    """
    raise IndexError(
        f"""Given labels are of size {y.shape} while they should be (n_samples,) \n"""
        + """If attempting multilabel classification, try using TabNetMultiTaskClassification """
        + """or TabNetRegressor"""
    )


# 이 딕셔너리는 다양한 분류 유형에 대해 고유한 라벨을 반환하는 함수를 매핑한것 
# "binary"와 "multicalss"는 _unique_multiclass 함수에 매핑되어 있고 
# "multilabel-indicator"는 _unique_indicator 함수에 매핑되어 있다 
_FN_UNIQUE_LABELS = {
    "binary": _unique_multiclass,
    "multiclass": _unique_multiclass,
    "multilabel-indicator": _unique_indicator,
}

# *ys는 가변 인자를 받아 여러개의 배열 또는 리스트를 입력으로 처리할 수 있음
# 이 함수는 입력에서 고유한 라벨을 추출하여 반환함
def unique_labels(*ys):
    """Extract an ordered array of unique labels

    We don't allow:
        - mix of multilabel and multiclass (single label) targets
        - mix of label indicator matrix and anything else,
          because there are no explicit labels)
        - mix of label indicator matrices of different sizes
        - mix of string and integer labels

    At the moment, we also don't allow "multiclass-multioutput" input type.

    Parameters
    ----------
    *ys : array-likes

    Returns
    -------
    out : numpy array of shape [n_unique_labels]
        An ordered array of unique labels.

    Examples
    --------
    >>> from sklearn.utils.multiclass import unique_labels
    >>> unique_labels([3, 5, 5, 5, 7, 7])
    array([3, 5, 7])
    >>> unique_labels([1, 2, 3, 4], [2, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> unique_labels([1, 2, 10], [5, 11])
    array([ 1,  2,  5, 10, 11])
    """
    
    # 입력된 인자가 없는 경우 에러를 발생, 입력값이 없는 상태에서 고유 라벨 추출할 수 없기 때문
    if not ys:
        raise ValueError("No argument has been passed.")
    # Check that we don't mix label format

    # type_of_target(x) 함수는 각 입력 데이터 x의 라벨 유형을 판별
    # 이 줄은 모든 입력 ys의 라벨 유형을 집합(set)으로 저장, 중복되는 유형은 제거 
    ys_types = set(type_of_target(x) for x in ys)
    
    # 입력 데이터의 라벨 유형이 "binary"와 "multiclass"로 섞여 있으면, 이를 "multiclass"로 통합
    # 이는 이 두 유형이 혼합되는 경우 허용되지 않기 때문에 처리 방식의 통일성을 유지하기 위함
    if ys_types == {"binary", "multiclass"}:
        ys_types = {"multiclass"}

    # 라벨 유형이 두 개 이상 섞여 있는 경우 허용되지 않으므로 에러를 발생시킴
    # 예를 들어, "binary"와 "multilabel-indicator"가 섞여 있는 경우 혼합 라벨 유형을 허용하지 않기 때문에 에러 발생
    if len(ys_types) > 1:
        raise ValueError("Mix type of y not allowed, got types %s" % ys_types)

    # ys_types에서 유일한 라벨 유형을 꺼내옴, 이후 이 유형을 기반으로 고유한 라벨 추출
    label_type = ys_types.pop()

    # Get the unique set of labels
    # 라벨 유형에 해당하는 고유 라벨을 추출하는 함수가 _FN_UNIQUE_LABELS 딕셔너리에서 선택
    # 예를 들어, "multiclass"라면 _unique_multiclass 함수가 반환
    _unique_labels = _FN_UNIQUE_LABELS.get(label_type, None)
    
    # 라벨 유형이 알려진 유형이 아닌 경우 에러를 발생시킴
    # 즉, _FN_UNIQUE_LABELS 에서 해당 라벨 유형에 대한 함수가 존재하지 않을 때 발생하는 에러 
    if not _unique_labels:
        raise ValueError("Unknown label type: %s" % repr(ys))

    # 각 입력 y에 대해 _unique_labels(y)를 호출하여 고유 라벨을 추출하고, 이를 모두 합친 뒤 집합(set)로 변환해 중복 라벨을 제거
    # chain.from_iterable()은 여러 리스트를 하나의 반복자로 결합해주는 역할
    ys_labels = set(chain.from_iterable(_unique_labels(y) for y in ys))

    # Check that we don't mix string type with number type
    # 문자열 라벨과 숫자 라벨이 혼합되었는지 확인
    # ys_labels 내의 모든 라벨이 문자열 타입인지, 아니면 숫자 타입인지를 체크하고, 두 타입이 섞여 있으면 에러를 발생시킴
    # 문자열과 숫자 라벨의 혼합은 허용되지 않기 때문
    if len(set(isinstance(label, str) for label in ys_labels)) > 1:
        raise ValueError("Mix of label input types (string and number)")

    # 최종적으로 고유 라벨들을 정렬한 후, Numpy 배열로 변환해 반환
    return np.array(sorted(ys_labels))


# y가 부동소수점(float)인지, 그리고 모든 값이 정수와 동일한 부동소수점 값인지 확인
# y.dtype.kind == "f" 는 y가 부동소수점(float)인지 확인
# y.astype(int) == y 는 소수점을 제거한 값과 원래 값이 동일한지 확인, 모두 동일하면 Trye 반환
def _is_integral_float(y):
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


# is_multilabel 함수는 입력 데이터 y가 멀티라벨 형식인지 확인 
def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.

    Parameters
    ----------
    y : numpy array of shape [n_samples]
        Target values.

    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    # y가 배열 속성(__array__)을 가지고 있으면 np.asarray(y)로 변환하여 NumPy 배열로 만듬 
    if hasattr(y, "__array__"):
        y = np.asarray(y)
        
    # 이 조건은 y가 2차원 배열이고, 두 번째 차원의 크기(y.shape[1])가 1보다 큰지 확인
    # 만약에 y가 2차원 배열이 아니거나, 두번째 차원이 1 이하인 경우 멀티라벨 형식이 아니므로 False 반환
    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    # y가 희소행렬(sparse matirx)인지 확인, 만약 희소 행렬이고 dok_matrix 또는 lil_matrix 형식이라면 이를 CSR 형식으로 변환(y.tocsr())
    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
            
        # 이 부분은 희소 행렬인 경우에만 해당 
        # y.data의 길이가 0이면 True를 반환 즉, 데이터가 없으면 멀티라벨 형식으로 간주 
        # np.unique(y.data).size == 1는 행렬에 고유한 값이 하나만 있는지 확인
        # 그리고 그 값이 부울형(b), 정수형(i) 또는 부호 없는 정수형(u) 이거나, 모든 값이 정수형 부동소수점 (_is_integral_float)이면 True를 반환
        return (
            len(y.data) == 0
            or np.unique(y.data).size == 1
            and (
                y.dtype.kind in "biu"
                or _is_integral_float(np.unique(y.data))  # bool, int, uint
            )
        )
        
    # 희소 행렬이 아닌경우, y에서 고유한 값을 추출하여 labels에 저장
    else:
        labels = np.unique(y)

        # labels의 고유한 값의 개수가 3 미만이어야 하고
        # y의 데이터 타입이 부울형,정수형,부호 없는 정수형이거나, 부동소수점이면서 모든 값이 정수와 동일해야 멀티라벨 형식으로 간주
        return len(labels) < 3 and (
            y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )

# 이 함수는 입력 y가 회귀 유형이 아닌지 확인하는 함수
def check_classification_targets(y):
    """Ensure that target y is of a non-regression type.

    Only the following target types (as defined in type_of_target) are allowed:
        'binary', 'multiclass', 'multiclass-multioutput',
        'multilabel-indicator', 'multilabel-sequences'

    Parameters
    ----------
    y : array-like
    """
    # type_of_target(y) 함수를 호출하여 y의 데이터 유형을 확인하고 이를 y_type에 저장
    y_type = type_of_target(y)
    
    # y_type 이 위에서 정의된 비회귀 유형이 아니면, ValueError를 발생시킨다
    # 비회귀 유형에는 이진, 다중클래스, 다중출력, 멀티라벨, 또는 멀티라벨 시퀀스만 허용
    if y_type not in [
        "binary",
        "multiclass",
        "multiclass-multioutput",
        "multilabel-indicator",
        "multilabel-sequences",
    ]:
        raise ValueError("Unknown label type: %r" % y_type)


# type_of_target 함수는 입력 데이터 y의 유형을 결정
def type_of_target(y):
    """Determine the type of data indicated by the target.

    Note that this type is the most specific type that can be inferred.
    For example:

        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.

    Parameters
    ----------
    y : array-like

    Returns
    -------
    target_type : string
        One of:

        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.

    Examples
    --------
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multiclass-multioutput'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    
    # y가 시퀀스 또는 희소 행렬(sparse matrix)인지, 또는 배열 속성(__array__)을 가지고 있는지 확인
    # y가 문자열이 아닌지도 확인, 이 조건을 만족해야 유효한 입력으로 간주한다 
    valid = (
        isinstance(y, (Sequence, spmatrix)) or hasattr(y, "__array__")
    ) and not isinstance(y, str)


    # y가 위의 조건을 만족하지 않으면 ValueError를 발생시킨다 즉, y가 배열 또는 문자열이 아닌 시퀀스가 아니면 에러가 발생 
    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), " "got %r" % y
        )

    # y가 SparseSeries 클래스인지 확인, SparseSeries는 허용되지 않으므로, 만약 그렇다면 ValueError를 발생시킴
    sparseseries = y.__class__.__name__ == "SparseSeries"
    if sparseseries:
        raise ValueError("y cannot be class 'SparseSeries'.")

    # y가 멀티라벨 형식인지 확인하고, 그렇다면 "multilabel-indicator"를 반환
    if is_multilabel(y):
        return "multilabel-indicator"

    # y를 Numpy 배열로 변환하려 시도, 실패하면 y의 유형을 "unknown" 으로 반환
    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return "unknown"


    # The old sequence of sequences format
    # y가 옛날 형식인 시퀀스의 시퀀스(sequence of sequences) 형식인지 확인
    # 시퀀스의 시퀀스는 더 이상 지원되지 않으므로, 만약 그렇다면 ValueError를 발생시킴
    # 예외 처리로 인덱스 에러가 발생하면 무시한다 
    try:
        if (
            not hasattr(y[0], "__array__")
            and isinstance(y[0], Sequence)
            and not isinstance(y[0], str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass


    # Invalid inputs
    # y가 3차원 이상의 배열이거나, 객체 타입이면서 첫 번째 요소가 문자열이 아닌 경우에는 y의 유형을 "unknown"으로 반환
    if y.ndim > 2 or (y.dtype == object and len(y) and not isinstance(y.flat[0], str)):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    # y가 2차원이지만 열의 개수가 0이면 y의 유형을 "unknown"으로 반환
    if y.ndim == 2 and y.shape[1] == 0:
        return "unknown"  # [[]]

    # y가 2차원이고 열의 개수가 1보다 크면, 멀티출력(multioutput) 형식이므로 
    # suffix에 "-multioutput"을 추가, 그렇지 않으면 빈 문자열을 할당 
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]


    # check float and contains non-integer float values
    # y가 부동 소수점(float) 타입이고, 값 중에 정수가 아닌 값이 있으면 continuous 타입으로 간주하고, suffix를 추가하여 반환
    if y.dtype.kind == "f" and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return "continuous" + suffix

    # 고유한 값의 개수가 2개보다 많으면 multiclass로 간주하고, 그렇지 않으면 binary로 간주하여 반환한다 
    if (len(np.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return "multiclass" + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]

# 입력 y를 판다스 시리즈로 변환한 후, 각 요소의 데이터 타입을 확인하고, 고유한 데이터 타입들을 target_types에 저장 
def check_unique_type(y):
    target_types = pd.Series(y).map(type).unique()
    
    # 타겟 데이터에 여러 타입이 섞여 있으면 에러를 발생시킨다
    # target_types의 길이가 1이 아니면, 즉, 다양한 타입이 존재하면 에러가 발생 
    if len(target_types) != 1:
        raise TypeError(
            f"Values on the target must have the same type. Target has types {target_types}"
        )


# 먼저 chech_unique_type(y_train) 함수를 호출하여 y_train이 동일한 데이터 타입을 가지는지 확인 
def infer_output_dim(y_train):
    """
    Infer output_dim from targets

    Parameters
    ----------
    y_train : np.array
        Training targets

    Returns
    -------
    output_dim : int
        Number of classes for output
    train_labels : list
        Sorted list of initial classes
    """
    check_unique_type(y_train)
    
    # unique_labels(y_train) 함수를 통해 y_train의 고유한 라벨들을 train_labels에 저장
    train_labels = unique_labels(y_train)
    
    # train_labels의 길이를 계산하여 출력 차원(output_dim)을 결정, 이는 고유한 클래스의 수를 의미 
    output_dim = len(train_labels)

    # 출력 차원과 고유한 클래스의 리스트를 반환
    return output_dim, train_labels


# 입력 y가 None이 아닌지 확인, 이 함수는 타겟 데이터 y가 훈련 시 사용된 라벨(labels)과 일치하는지 확인하는 함수 
def check_output_dim(labels, y):
    if y is not None:
        
        # 먼저 y가 동일한 타입의 데이터들로 구성되어 있는지 확인 
        check_unique_type(y)
        
        # unique_labels(y)를 사용해 y의 고유한 라벨들을 valid_labels에 저장
        valid_labels = unique_labels(y)
        
        # valid_labels 가 훈련된 라벨(labels)의 부분 집합인지 확인한다 부분 집합이 아니라면 에러가 발생 
        if not set(valid_labels).issubset(set(labels)):
            
            # 타겟에 훈련되지 않은 라벨이 포함되어 있으면 Value Error를 발생시키고, 적절한 에러 메시지를 출력
            raise ValueError(
                f"""Valid set -- {set(valid_labels)} --
                             contains unkown targets from training --
                             {set(labels)}"""
            )
    return


# y_train의 차원이 2보다 작은지 확인, 다중 작업에서는 y_train이 (n_examples, n_tasks)형식이어야 함
def infer_multitask_output(y_train):
    """
    Infer output_dim from targets
    This is for multiple tasks.

    Parameters
    ----------
    y_train : np.ndarray
        Training targets

    Returns
    -------
    tasks_dims : list
        Number of classes for output
    tasks_labels : list
        List of sorted list of initial classes
    """

    if len(y_train.shape) < 2:
        
        # y_train이 잘못된 형식을 가지면 에러 메시지와 함께 ValueError를 발생시킴
        raise ValueError(
            "y_train should be of shape (n_examples, n_tasks)"
            + f"but got {y_train.shape}"
        )
    
    # y_train의 두 번째 차원인 작업(task)의 수를 nb_tasks에 저장
    nb_tasks = y_train.shape[1]
    
    # 각 작업의 출력 차원(tasks_dims)과 라벨(tasks_labels)을 저장할 빈 리스트를 생성 
    tasks_dims = []
    tasks_labels = []
    
    # 작업의 수만큼 반복문을 실행
    for task_idx in range(nb_tasks):
        
        # 각 작업의 타깃 데이터를 y_train[:, task_idx]로 추출하여 infer_output_dim 함수를 통해 
        # 출력차원 (output_dim)과 고유한 라벨들(train_labels)을 계산
        try:
            output_dim, train_labels = infer_output_dim(y_train[:, task_idx])
            
            # 계산된 output_dim과 train_labels를 각각 tasks_dim와 tasks_labels에 추가
            tasks_dims.append(output_dim)
            tasks_labels.append(train_labels)
            
        # 만약 작업에서 오류가 발생하면, 해당 작업 인덱스와 함께 에러 메시지를 출력하면서 ValueError를 발생시킴
        except ValueError as err:
            raise ValueError(f"""Error for task {task_idx} : {err}""")
        
    # 모든 작업의 출력 차원(tasks_dims)과 라벨(tasks_labels)을 반환 
    return tasks_dims, tasks_labels
