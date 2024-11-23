#=================================================================
#   Training_FMA_DenseNet169.py
# Music Genre Classification Using DenseNet and Data Augmentation
# Authors: TV Loan,DTL Thuy
#=================================================================
#------------------------------------------------------------------
# Version 19/7/2021, running on GPU 2080, saving figures on PNG files
# Python 3.8, Keras: 2.4.3, tensorflow: 2.2, tensorflow-gpu: 2.3 ,
# using interp from numpy (not from scipy)
# saving report file
#-------------------------------------------------------------------
"""import libs"""
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Flatten
import numpy as np 
import pandas as pd 
import glob
import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings('ignore')
import datetime

import csv, ast
import pickle


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201

#-------------------------------------------------------------------
# GPU Detection
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))

tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)

number_class = 8 # 8 Music Genres
batch_sz = 32
nepoch = 500
add_info = "Noise_Echo_FMA"
lr_min=0.00001
str_lr_min = "1e-5"

# Path to the *.CSV file -> the list of files for training and validation
train_data = pd.read_csv("./FMA_trainvalid_data.csv")

# Verify if CSV file is qualified
for i in range(train_data.genre_name.shape[0]):
    train_data.genre_name.loc[i] = ast.literal_eval(train_data.genre_name[i])


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K
"""

---

Create data generator

"""

# Path to directory containing image files *.png
train_img_path = "./FMA_IMAGES/"
# Create subdirectory for 9 folds
for k in range (1,10):
    sub_dir = "CRV" + str(k)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

# 8 music genres of FMA
name_class = ["Electronic",
            "Experimental",
            "Folk",
            "Hip_Hop",
            "Instrumental",
            "International",
            "Pop",
            "Rock"
    ]
def data_generator(train, val):
    train_datagen = ImageDataGenerator(rescale=1/255.,)
    val_datagen = ImageDataGenerator(rescale=1/255.,)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=True,
        class_mode='categorical',
        classes=name_class

    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=train_img_path,
        x_col="song_id",
        y_col="genre_name",
        batch_size=batch_sz,
        target_size=(224,224),
        shuffle=False,
        class_mode='categorical',
        classes=name_class
    )   
    return train_generator, val_generator


def train_model(model, train_gen, val_gen, train_steps, val_steps, epochs, fold):
    callbacks = [
        EarlyStopping(patience=50, verbose=1, mode='auto'),
        ReduceLROnPlateau(factor=0.1, patience=2, min_lr=lr_min, verbose=1, mode='auto'),
        ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True, mode='auto', verbose=1)
    ]

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                metrics = ['binary_accuracy'])

    working_dir = "./CRV" + str(fold) + "/figures/"
    # figures_dir = working_dir + "CRV" + str(fold) + "/"
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    history_dir = "./history"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    from contextlib import redirect_stdout
    # Store model architecture in text file, it is a very long file for DenseNet
    with open('./modelNewDenseNet169.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    model.summary()
    file_txt = working_dir + "time169.txt"
    x1 = datetime.datetime.now()
    time_now_begin = x1.strftime("%d%b%y%a") + "_" + x1.strftime("%I%p%M") + "\n"
    # with open(file_txt, 'a') as f:
    #     f.write(time_now_begin)
    H = model.fit(train_gen,
            steps_per_epoch=train_steps,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_gen,
            validation_steps=val_steps,
            verbose=1)

    with open('./history/fold{}.pickle'.format(fold), 'wb') as file_pi:
        pickle.dump(H.history, file_pi)
    # visualizing losses and accuracy
    train_loss = H.history['loss']
    train_acc = H.history['binary_accuracy']
    val_loss = H.history['val_loss']
    val_acc = H.history['val_binary_accuracy']
    xc = range(len(train_loss))
    x2 = datetime.datetime.now()
    # time_now_end = x2.strftime("%d%b%y%a") + "_" + x2.strftime("%I%p%M") + "\n"
    # number_epoch = len(train_loss)
    # with open(file_txt, 'a') as f:
    #     f.write(time_now_end)
    #     f.write(" Number of Epochs = " + str(number_epoch))
    figures_dir = working_dir
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    plt.subplot(211)
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train_loss", "val_loss"], loc="upper right")
    plt.subplot(212)

    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train_acc", "val_acc"], loc="lower right")
    max_val_acc = max(val_acc)
    plt.title("Max Validation Accuracy = " + str(max_val_acc))
    x = datetime.datetime.now()
    time_now = x.strftime("%d%b%y%a") + "_" + x.strftime("%I%p%M")
    fig_filename1 = figures_dir + "LossAcc"+"-"+str(batch_sz)+"_"+str_lr_min+"_"+add_info+"_"+time_now
    fig.savefig(fig_filename1)

# Begin Training and Validation for 9 folds
fold = 1
kfold = KFold(n_splits=9, shuffle=True,random_state=42)
for train_idx, val_idx in kfold.split(train_data):
    print('=========================='*5)
    print('Fold', fold)
    train, val = train_data.loc[train_idx], train_data.loc[val_idx]
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    train_generator, val_generator = data_generator(train, val)
    # Using DenseNet
    base_model_densenet = DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    headModel = base_model_densenet.output
    headModel = Dropout(0.5)(headModel)
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(1024, activation='relu')(headModel)
    headModel = Flatten()(headModel)
    headModel = Dense(1024, activation='relu')(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(1024, activation='relu')(headModel)
    headModel = Dense(number_class, activation='sigmoid')(headModel)
    model_dense_121 = Model(inputs=base_model_densenet.input, outputs=headModel)
    # File path for storing model weights
    file_path="./weights/fold_{}_best_weight.h5".format(fold)
    train_steps = int(len(train)/batch_sz)
    val_steps = int(len(val)/batch_sz)

    train_model(model_dense_169, train_generator, val_generator, train_steps, val_steps, nepoch, fold)
    print('========='*10)
    # Reset for the next fold
    del model_dense_169
    K.clear_session()
    fold += 1

