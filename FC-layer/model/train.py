import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import FmaMLP  # 모델 정의가 포함된 파일

# 데이터 로드 함수
def load_data(input_folder, target_path, excluded_ids):
    X_list = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npz'):
            file_path = os.path.join(input_folder, file_name)
            input_data = np.load(file_path)
            if 'mfcc' in input_data:
                mfcc = input_data['mfcc']
                reshaped_mfcc = mfcc.reshape(mfcc.shape[0], -1)
                X_list.append(reshaped_mfcc)

    X = np.concatenate(X_list, axis=0) if X_list else np.array([])

    target_data = np.load(target_path)
    track_ids = target_data['track_id']
    y = target_data['genre']  # 원-핫 인코딩된 상태로 y 유지

    print(f"Loaded Track IDs: {track_ids[:5]}")
    print(f"Loaded Genres: {y[:5]}")

    valid_indices = [i for i, track_id in enumerate(track_ids) if track_id not in excluded_ids]
    X = X[valid_indices] if X.size > 0 else X
    y = y[valid_indices]

    return X, y

# 제외할 track_id 리스트
excluded_track_ids = {"098565", "098567", "098569", "099134", "108925", "133297"}

# 데이터 로드
input_npz_folder = '/home/hwang-gyuhan/Workspace/Transformer/mfcc_segments'  # 입력 폴더 경로 지정
target_npz_path = '/home/hwang-gyuhan/Workspace/Transformer/preprocessing/track_genre_data.npz'  # 타겟 파일 경로 지정
X, y = load_data(input_npz_folder, target_npz_path, excluded_track_ids)

# X와 y의 크기 확인
if X is not None and y is not None:
    print(f"X의 샘플 수: {X.shape[0]}")
    print(f"y의 샘플 수: {y.shape[0]}")

    num_samples = X.shape[0]
    if num_samples == y.shape[0]:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]

        train_size = int(0.8 * num_samples)
        valid_size = int(0.1 * num_samples)

        X_train, X_temp = X[:train_size], X[train_size:]
        y_train, y_temp = y[:train_size], y[train_size:]

        X_valid, X_test = X_temp[:valid_size], X_temp[valid_size:]
        y_valid, y_test = y_temp[:valid_size], y_temp[valid_size:]

        print(f'Train Shape: {X_train.shape}, {y_train.shape}')
        print(f'Validation Shape: {X_valid.shape}, {y_valid.shape}')
        print(f'Test Shape: {X_test.shape}, {y_test.shape}')

        # PyTorch Tensor로 변환
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_valid, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(np.argmax(y_valid, axis=1), dtype=torch.long).to(device) 
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long).to(device)  

        # 모델 학습 코드 계속...
        hyperparams = FmaMLP.get_hyperparameters()
        model = FmaMLP(hyperparams['num_classes'], hyperparams['dropout']).to(device)
        optimizer = optim.Adam(model.parameters())

        num_epochs = hyperparams['num_epochs']
        batch_size = hyperparams['batch_size']

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.NLLLoss()(outputs, targets)  # NLLLoss 사용
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)

            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = nn.NLLLoss()(outputs, targets)
                    running_val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_val += targets.size(0)
                    correct_val += (predicted == targets).sum().item()

            val_loss = running_val_loss / len(val_loader)
            val_losses.append(val_loss)
            val_accuracy = 100 * correct_val / total_val
            val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

        # 결과 시각화
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='orange')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()