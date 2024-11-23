"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # 위치 인코딩을 초기화
        # max_len x d_model 크기의 영행렬 생성 
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
    # self.encoding
    # [max_len = 512, d_model = 512]

        batch_size, seq_len, d_model = x.size()
    # [batch_size = 128, seq_len = 250, d_model = 512]

    # 위치 인코딩을 seq_len에 맞게 선택
        positional_encoding = self.encoding[:seq_len, :].to(x.device)
    
    # 배치 크기에 맞게 위치 인코딩 확장
        positional_encoding = positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]

    # 위치 인코딩을 입력에 더하기
        positional_embedding = x + positional_encoding

        return positional_embedding