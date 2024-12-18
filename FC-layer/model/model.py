import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 정의
class FmaMLP(nn.Module):
    def __init__(self, num_class, drop_prob):
        super(FmaMLP, self).__init__()

        self.dropout = nn.Dropout(p=drop_prob)
        self.linear1 = nn.Linear(3250, 4096)  # 입력 차원 변경
        self.linear2 = nn.Linear(4096, 8192)
        self.linear3 = nn.Linear(8192, 8192)
        self.linear4 = nn.Linear(8192, 4096)
        self.linear5 = nn.Linear(4096, 2048)
        self.linear6 = nn.Linear(2048, 1024)
        self.linear7 = nn.Linear(1024, 512)
        self.linear8 = nn.Linear(512, 256)
        self.linear9 = nn.Linear(256, 64)

        self.reduce_layer = nn.Linear(64, num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.float()

        mlp1 = F.relu(self.linear1(x))
        mlp1 = self.dropout(mlp1)
        mlp2 = F.relu(self.linear2(mlp1))
        mlp2 = self.dropout(mlp2)
        mlp3 = F.relu(self.linear3(mlp2))
        mlp3 = self.dropout(mlp3)
        mlp4 = F.relu(self.linear4(mlp3))
        mlp4 = self.dropout(mlp4)
        mlp5 = F.relu(self.linear5(mlp4))
        mlp5 = self.dropout(mlp5)
        mlp6 = F.relu(self.linear6(mlp5))
        mlp6 = self.dropout(mlp6)
        mlp7 = F.relu(self.linear7(mlp6))
        mlp7 = self.dropout(mlp7)
        mlp8 = F.relu(self.linear8(mlp7))
        mlp8 = self.dropout(mlp8)
        mlp9 = F.relu(self.linear9(mlp8))
        mlp9 = self.dropout(mlp9)
        
        output = self.reduce_layer(mlp9)
        return self.logsoftmax(output)

    @staticmethod
    def get_hyperparameters():
        return {
            "dropout": 0.3,
            "num_classes": 8,
            "learning_rate": 1e-5,
            "num_epochs": 400,
            "batch_size": 128,
        }