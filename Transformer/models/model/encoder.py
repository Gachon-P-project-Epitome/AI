"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import sys
sys.path.append('/home/hwang-gyuhan/Workspace/Transformer')
from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding




class Encoder(nn.Module):

    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, num_classes, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        
        # 인코더 출력 후 분류를 위한 Linear 레이어 추가
        self.fc = nn.Linear(d_model, num_classes)  # d_model 차원에서 num_classes로 출력

    def forward(self, x):
        x = self.emb(x)

        # 인코더 레이어 통과
        for layer in self.layers:
            x = layer(x)  

        x = x.mean(dim=1)  # (batch_size, seq_len, d_model)에서 seq_len 차원을 평균내어 (batch_size, d_model)로 변환
        x = self.fc(x)

        return x