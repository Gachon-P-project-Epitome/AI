#=================================================================
#   Testing_FMA_DenseNet121.py
# Music Genre Classification Using DenseNet and Data Augmentation
# Authors: TV Loan, DTL Thuy
#=================================================================
"""import libs"""
import itertools
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import warnings
warnings.filterwarnings('ignore')
import datetime
import csv, ast
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_auc_score
import scikitplot as skplt
from sklearn import metrics

# Path to CSV file containing the list of files for testing
test_data = pd.read_csv("/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/test_data.csv")
sample_number = len(test_data.song_id)

# Number of music genres for FMA
number_class = 8
batch_sz = 32

# Verify if CSV file is qualified
for i in range(test_data.genre_name.shape[0]):
    test_data.genre_name.loc[i] = ast.literal_eval(test_data.genre_name[i])

# 8 genres for FMA
name_class = ["Electronic",
            "Experimental",
            "Folk",
            "Hip_Hop",
            "Instrumental",
            "International",
            "Pop",
            "Rock"]

tags = np.array(name_class)

test_data_gen = ImageDataGenerator(rescale=1/255.,)

# Path to the directory containing image files
test_img_path = "/home/hwang-gyuhan/Workspace/dataset/FMA_IMAGES/"
test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test_data,
        directory=test_img_path,
        x_col='song_id',
        y_col='genre_name',
        batch_size=batch_sz,
        target_size=(224, 224),
        class_mode='categorical',
        classes=name_class,
        shuffle=False)

def get_y_pred(avg_result):
    y_pred = []
    for sample in avg_result:
        toto = np.max(sample)
        y_pred.append([1.0 if (i == toto) else 0.0 for i in sample])
    y_pred = np.array(y_pred)
    return y_pred

# Test single weights file
weights_file_path = '/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing//weights/epoch_010_weights.h5'  # 테스트할 가중치 파일 경로 지정

# 모델 생성 및 가중치 로드
base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
headModel = base_model_densenet.output
headModel = Dropout(0.5)(headModel)
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Flatten()(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1024, activation='relu')(headModel)
headModel = Dense(8, activation='sigmoid')(headModel)  # Assuming 8 classes for music genres
model_dense_121 = Model(inputs=base_model_densenet.input, outputs=headModel)

# 가중치를 로드
model_dense_121.load_weights(weights_file_path)

# 테스트 수행
figures_dir = "/home/hwang-gyuhan/Workspace/DenseNet/ForTraining&Testing/figures/epoch_010_test"
result = model_dense_121.predict(test_generator)
y_pred_res = get_y_pred(result)
y_pred = np.where(y_pred_res == 1)[1]
real_labels_frames = test_generator.labels
y_real_label = np.array(real_labels_frames)
y_real_label = y_real_label.flatten()
cnf_matrix_org = confusion_matrix(y_real_label, y_pred)

# Original Confusion Matrix-------------------------------
fig = plt.figure(2)
plt.subplots_adjust(top=0.5)
plt.subplots_adjust(bottom=0.1)
cmap = plt.cm.get_cmap('Blues')
plt.imshow(cnf_matrix_org, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(tags))
plt.xticks(tick_marks, tags, rotation=45)
plt.yticks(tick_marks, tags)
thresh = cnf_matrix_org.max() / 2.
for i, j in itertools.product(range(cnf_matrix_org.shape[0]), range(cnf_matrix_org.shape[1])):
    plt.text(j, i, cnf_matrix_org[i, j],
             horizontalalignment="center",
             color="white" if cnf_matrix_org[i, j] > thresh else "black")

plt.tight_layout()
title = "Confusion Matrix"
x = datetime.datetime.now()
time_now = x.strftime("%d%b%y%a") + "_" + x.strftime("%I%p%M") + "_"
model_name_from_file = os.path.basename(weights_file_path).replace('.h5', '')

plt.ylabel('True label')
plt.xlabel('Predicted label')
file_name = os.path.join(figures_dir, model_name_from_file + "_" + time_now + "CFM_ORG")
fig.savefig(file_name, bbox_inches='tight')
plt.clf()
np.savetxt(file_name + ".txt", cnf_matrix_org, fmt="%d")

# 추가된 코드: 각 노래별 예측된 장르 저장
output_file = os.path.join(figures_dir, "predicted_genres.csv")
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['song_id', 'actual_genre', 'predicted_genre'])
    for idx, song_id in enumerate(test_data['song_id']):
        actual_genre = test_data['genre_name'][idx]
        predicted_genre = name_class[y_pred[idx]]  # 예측된 장르 이름
        writer.writerow([song_id, actual_genre, predicted_genre])

# ROC curve-------------------------------
axes = skplt.metrics.plot_roc_curve(y_real_label, y_pred_res)
fig_filename1 = os.path.join(figures_dir, model_name_from_file + "_" + time_now + "_ROC")
axes.figure.savefig(fig_filename1)

auc_ = round(roc_auc_score(
    y_real_label,
    y_pred_res,
    average='macro',
    sample_weight=None,
    max_fpr=None,
    multi_class='ovr',
    labels=None
), 3)

file_report = os.path.join(figures_dir, model_name_from_file + "_" + time_now + "_REPORT.txt")
print(metrics.classification_report(y_real_label, y_pred, digits=4),
      file=open(file_report, "a"))
print("AUC             ", auc_, file=open(file_report, "a"))