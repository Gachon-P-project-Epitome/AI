import numpy as np
import os
import sys
sys.path.append('/home/hwang-gyuhan/Workspace/Transformer')
from conf import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset



class Preprocessing:

    def __init__(self,
                 input_npz_folder='/home/hwang-gyuhan/Workspace/Transformer/mfcc_segments',
                 target_npz_path='/home/hwang-gyuhan/Workspace/Transformer/preprocessed_data_with_labels.npz'
                ):
       
        self.input_npz_folder = input_npz_folder 
        self.target_npz_path = target_npz_path
        self.X = np.array([])
        self.y = np.array([])
        self.track_ids = np.array([])  

    def load_data(self):
       
        X_list = []

        if not os.listdir(self.input_npz_folder):
            print("입력 NPZ 폴더가 비어 있습니다.")
            return None, None, None, None, None, None

        for file_name in os.listdir(self.input_npz_folder):
            if file_name.endswith('.npz'):
                file_path = os.path.join(self.input_npz_folder, file_name)
                input_data = np.load(file_path)

                if 'mfcc' in input_data:
                    mfcc = input_data['mfcc']
                    X_list.append(mfcc)

        self.X = np.concatenate(X_list, axis=0) if X_list else np.array([])

        target_data = np.load(self.target_npz_path)
        self.track_ids = target_data['track_id']  
        self.y = target_data['genre']

        valid_indices = [i for i, track_id in enumerate(self.track_ids)]
        self.X = self.X[valid_indices] if self.X.size > 0 else self.X
        self.y = self.y[valid_indices]
        self.track_ids = self.track_ids[valid_indices]  

        if self.X.size == 0 or self.y.size == 0:
            print("데이터가 로드되지 않았습니다.")
            return None, None, None, None, None, None
        


        X_train, X_temp, y_train, y_temp, track_ids_train, track_ids_temp = train_test_split(
            self.X, self.y, self.track_ids, test_size=0.2, random_state=42, stratify=self.y)
        X_valid, X_test, y_valid, y_test, track_ids_valid, track_ids_test = train_test_split(
            X_temp, y_temp, track_ids_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        X_train = X_train.transpose((0, 2, 1))  # (6395, 20, 250) -> (6395, 250, 20)
        X_valid = X_valid.transpose((0, 2, 1))  # (799, 20, 250) -> (799, 250, 20)
        X_test = X_test.transpose((0, 2, 1))    # (800, 20, 250) -> (800, 250, 20)
        

        print("X_train 형태:", X_train.shape)
        print("y_train 형태:", y_train.shape)
        print("X_valid 형태:", X_valid.shape)
        print("y_valid 형태:", y_valid.shape)
        print("X_test 형태:", X_test.shape)
        print("y_test 형태:", y_test.shape)
        
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
        y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f'Train Loader: {len(train_loader)} batches')
        print(f'Valid Loader: {len(valid_loader)} batches')
        print(f'Test Loader: {len(test_loader)} batches')
    
        return train_loader, valid_loader, test_loader


