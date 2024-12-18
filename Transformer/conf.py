"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cpu")

# model parameter setting
batch_size = 64
max_len = 256
d_model = 20
n_layers = 6
n_heads = 10
ffn_hidden = 640
drop_prob = 0.4
num_classes = 8

# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 200
clip = 1.0
weight_decay = 5e-4
inf = float('inf')