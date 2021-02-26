from memory_profiler import memory_usage
import os
import pandas as pd
import time
from glob import glob
import numpy as np
from PIL import Image
from keras import layers
from keras import models
import math
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import keras.backend as K
import librosa
import librosa.display
import pylab
import csv
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType

def test_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'C:/Users/srika/PycharmProjects/Classifier/kaggle/working/experiment_spectrogram/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'C:/Users/srika/PycharmProjects/Classifier/kaggle/working/train/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'C:/Users/srika/PycharmProjects/Classifier/kaggle/working/test/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


def data_split(path_to_data, img_size=256,label_csv='train.csv'):

    training_list = np.stack([np.asarray(Image.open(l).convert('RGB').resize((img_size, img_size))) for l in path_to_data], axis=0)
    id_list=[]
    class_id_list=[]
    class_list=[]
    train_data = pd.read_csv(label_csv, header=0)
    class_dict = {}
    for i in path_to_data:
        res='.'.join(i.split('.')[:-2])
        fin=os.path.basename(res)
        var=train_data.loc[train_data['ID']==fin]
        class_id=var['Class'].tolist()
        label=var['class'].tolist()
        id_list.append(fin)
        if class_id[0] not in class_dict:
            class_dict[class_id[0]] = label[0]
        try:
            class_id_list.append(class_id[0])
            class_list.append(label[0])
        except:
            print(fin)
            print(i,class_id,label)
    label_arr=np.array(class_id_list).astype(int)
    return training_list, label_arr,class_dict


def main():
    #After creating the spectrograms, comment out lines 115-130
    Data_dir=np.array(glob("C:/Users/srika/PycharmProjects/Classifier/train/*"))
    print(Data_dir)
    Data_dir_test = np.array(glob("C:/Users/srika/PycharmProjects/Classifier/test/*"))
    i=0
    for file in Data_dir[i:i+8000]:
        filename,name = file,file.split('\\')[-1]
        print(filename)
        print(name)
        create_spectrogram(filename,name)

    j = 0
    for file in Data_dir_test[j:j+2000]:
        filename,name = file,file.split('\\')[-1]
        print(filename)
        print(name)
        create_spectrogram_test(filename, name)

    gc.collect()

    train_images_list=np.array(glob("C:/Users/srika/PycharmProjects/Classifier/kaggle/working/train/*"))
    test_images_list = np.array(glob("C:/Users/srika/PycharmProjects/Classifier/kaggle/working/test/*"))
    data_split(train_images_list, 32)
    csv_file='train.csv'
    csv_file_2='test.csv'
    train_images, train_labels,classes = data_split(train_images_list, 32,csv_file)
    test_images, test_labels,classes = data_split(test_images_list, 32,csv_file_2)
    train_mean = np.mean(np.mean(np.mean(train_images, axis=0), axis=0), axis=0)

    np.savez_compressed('file.npz', train_mean=train_mean)

    print("Classes: ")
    print(classes)
    train_images, test_images = (train_images - train_mean[None,None,None,:]) / 256.0, (test_images -  train_mean[None,None,None,:]) / 256.0
    print(train_images.max(), test_images.max())
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(11))

    model.summary()

    #opt = Adam(learning_rate=0.01)
    opt = SGD(learning_rate=0.001)

    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    history = model.fit(train_images, train_labels, batch_size=5, epochs=100, validation_data=(test_images, test_labels))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.4, 1])
    plt.title('Validation Curve')
    plt.legend(loc='lower right')
    plt.savefig('Validation_curve.png')
    plt.show(block=False)
    plt.pause(5)
    plt.close("all")
    #Register the model
    model.save('spectrogram_save.h5')

    #Model Prediction section
    test_dir = np.array(glob("C:/Users/srika/PycharmProjects/Classifier/kaggle/working/experiment/*"))
    i = 0
    actual_result_list=[]
    filename_list=[]
    train_data = pd.read_csv('train.csv', header=0)
    for file in test_dir[i:i+50]:
        filename,name = file,file.split('\\')[-1]
        filename_list.append(name)
        if ord(name[0]) in range(ord('A'), ord('M')):
            result='Elephant'
            actual_result_list.append(result)
        else:
            res = file.split('.')[0]
            fin = os.path.basename(res)

            var = train_data.loc[train_data['ID'] == fin]
            label = var['class'].tolist()
            actual_result_list.append(label[0])
    test_spectrogram(filename,name)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    test= np.array(glob("C:/Users/srika/PycharmProjects/Classifier/kaggle/working/experiment_spectrogram/*"))
    img_size=32
    Input = np.stack([np.asarray(Image.open(l).convert('RGB').resize((img_size, img_size))) for l in test], axis=0)
    #Predicting the model
    output = model.predict((Input - train_mean[None,None,None,:])/256.)
    preds = [np.argmax(l) for l in output]
    cls_preds = [classes[l] for l in preds]
    print("================================")
    print("------MODEL PREDICTIONS---------")
    print("================================")
    final=list(zip(filename_list,actual_result_list,cls_preds))
    compare=list(zip(actual_result_list,cls_preds))
    result_df=DataFrame(final,columns=['Filename','Actual_Result','Model_Predicted_Result'])
    print(result_df)
    correct_count=0
    incorrect_count=0
    for row in compare:
        actual=row[0]
        predicted=row[1]
        if actual == predicted:
            correct_count+=1
        else:
            incorrect_count+=1
    total=correct_count+incorrect_count
    pct = (float(correct_count)/(correct_count+incorrect_count))*100
    significant_digits = 4
    rounded_number = round(pct, significant_digits - int(math.floor(math.log10(abs(pct)))) - 1)
    print("This model has predicted "+str(correct_count)+"/"+str(total)+" correctly, which is "+str(rounded_number)+"%")

if __name__ == '__main__':
    start_time=time.time()
    main()
    end_time=time.time()
    del_t=end_time-start_time
    print("Time of execution: ",del_t)