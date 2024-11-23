import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/hwang-gyuhan/Workspace/Transformer')
from data import *
from conf import *
from models.model.transformer import Encoder
from util.epoch_timer import epoch_time

# 모델 파라미터 개수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 가중치 초기화
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)

# 학습 결과를 그래프로 저장
def save_training_plot(train_losses, valid_losses, train_accuracies, valid_accuracies, save_path='training_plot.png'):
    plt.figure(figsize=(12, 8))

    # Loss 그래프
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy 그래프
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(valid_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 모델 초기화
model = Encoder(
    d_model=d_model,
    max_len=max_len,
    num_classes=num_classes,
    ffn_hidden=ffn_hidden,
    n_head=n_heads,
    n_layers=n_layers,
    drop_prob=drop_prob,
    device=device
).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)

# Optimizer와 Scheduler 설정
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 데이터 로드
data_loader = Preprocessing()
train_loader, valid_loader, test_loader = data_loader.load_data()

# 학습 함수
def train(model, train_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for i, batch in enumerate(train_loader):
        src = batch[0].to(device)
        trg = batch[1].to(device)

        optimizer.zero_grad()
        output = model(src)
        output_reshape = output.contiguous().view(-1, output.shape[-1])

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        preds = output_reshape.argmax(dim=1)
        correct += (preds == trg).sum().item()
        total += trg.size(0)

        print(f'Step: {round((i / len(train_loader)) * 100, 2)}% , Loss: {loss.item()}')

    accuracy = correct / total
    return epoch_loss / len(train_loader), accuracy

# 평가 함수
def evaluate(model, valid_loader, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            src = batch[0].to(device)
            trg = batch[1].to(device)

            output = model(src)
            output_reshape = output.contiguous().view(-1, output.shape[-1])

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            preds = output_reshape.argmax(dim=1)
            correct += (preds == trg).sum().item()
            total += trg.size(0)

    accuracy = correct / total
    return epoch_loss / len(valid_loader), accuracy

# 학습 실행
def run(total_epoch, best_loss):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for step in range(total_epoch):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%')
        print(f'\tVal Loss: {valid_loss:.3f} | Val Acc: {valid_acc * 100:.3f}%')

    save_training_plot(
        train_losses, valid_losses, train_accuracies, valid_accuracies,
        save_path='training_results.png'
    )
    print('Training results have been saved to "training_results.png".')

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)