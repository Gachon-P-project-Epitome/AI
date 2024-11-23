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

import warnings
warnings.filterwarnings('ignore')
import datetime
import csv, ast
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#-------------------------------------------------------------------
#MORE-----------------------------------------------------
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import scikitplot as skplt
from sklearn import metrics
#-------------------------------------------------------------------
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)

# Path to CSV file containing the list of files for testing
test_data = pd.read_csv("./FMA_test_data.csv")
sample_number = len(test_data.song_id)

# Number of mugic genres for FMA
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
test_img_path = "./FMA_IMAGES/"
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

models_resnet = []
# Weights of model saved from Training
weights_path = './weights/'

 # Load model weights from Training
for file_name in os.listdir(weights_path):
    model = keras.models.load_model(os.path.join(weights_path, file_name))
    models_densenet.append(model)

from sklearn.metrics import confusion_matrix



test_sample = sample_number
y_pred_res = np.zeros((test_sample, number_class))

# Testing for 9 folds
fold= 1
model_name = "DenseNet121"
for model in models_densenet:
    working_dir = "./" +  "CRV" + str(fold) +"/figures/"
    figures_dir = working_dir
    result = model.predict(test_generator)
    y_pred_res = get_y_pred(result)
    y_pred = np.where(y_pred_res == 1)[1]
    real_labels_frames = test_generator.labels
    y_real_label = np.array(real_labels_frames)
    y_real_label = y_real_label.flatten()
    cnf_matrix_org = confusion_matrix(y_real_label, y_pred)
    # Original Confusion Matric------------------------------
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
    title = "Confusion Matrix " + "CRV" + str(fold) # , "+"Average Test Accuracy = " + str(average_acc) + " %"#, Max Epoch = "+str_number
    plt.title(title)
    x = datetime.datetime.now()
    time_now = x.strftime("%d%b%y%a") + "_" + x.strftime("%I%p%M") + "_"
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    file_name = figures_dir + "/" + time_now + "CFM_ORG"  # +'{:.2e}'.format(average_acc).replace(".", "")# + "_MxEp_"+str_number
    # Saving confusion matrix
    fig.savefig(file_name, bbox_inches='tight')
    plt.clf()
    np.savetxt(file_name + ".txt", cnf_matrix_org, fmt="%d")  # ,fmt="%6.2f")
    axes = skplt.metrics.plot_roc_curve(y_real_label,y_pred_res )
    fig_filename1 = figures_dir + "/" + time_now + "_" + model_name + "_ROC"  # +'{:.2e}'.format(args.lr_decay)
    # Saving ROC figure
    axes.figure.savefig(fig_filename1)
    # overall_accuracy = round(100.0 * accuracy_score(y_real_label, y_pred), 2)
    # print("Scikit overall_accuracy = ", overall_accuracy)
    auc_ = round(roc_auc_score(
        y_real_label,
        y_pred_res ,
        average='macro',
        sample_weight=None,
        max_fpr=None,
        multi_class='ovr',
        labels=None
    ), 3)
    # All the things is stored here, in file_report
    file_report = figures_dir + "/" + "CRV" + str(fold) + "_4digits_REPORT_DenseNet121.txt"
    print(metrics.classification_report(y_real_label, y_pred, digits=4),
          file=open(file_report, "a"))
    # Adding AUC to report file
    print("AUC             ", auc_, file=open(file_report, "a"))
    fold = fold + 1