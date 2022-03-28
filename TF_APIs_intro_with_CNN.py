# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:18:20 2021

@author: Akshay Khot
"""

"""
Objective: 
    To understand the different APIs for the network development in tensorflow
    1. Sequential API
    2. Functional API
    Both APIs are applied in context of CNN
"""

"""
some temp dev branch is created
"""
"""
some other changes are made.
"""

#%% import libraries
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import time


#%% import dataset
from keras.datasets import mnist
(x_train, y_train), (x_test,y_test)=mnist.load_data()
#(x_train, y_train), (x_test,y_test)=tf.keras.datasets.cifar10.load_data()

#understand data size and shape
print("x_train : ",x_train.shape)
print("y_train : ",y_train.shape)
print("x_test  : ",x_test.shape)
print("y_test  : ",y_test.shape)
#list_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#visualize the image
print('Before normalization : Min={}, max={}'.format(x_train.min(),x_train.max()))

# Normalise your data (x_train and x_test) to bring the pixel values between 0 and 1
# write your code here
x_train = x_train/255.0
x_test = x_test/255.0

print('After normalization  : Min={}, max={}'.format(x_train.min(),x_train.max()))

# Visualization
"""
f, axs =plt.subplots(1, 2,figsize=(2,2) )

for i in range(2):
    n=random.choice(list_names)
    p=list_names.index(n)
    print(n,p)
    image=x_train[p]   #selecting the images where y_label is same as int n
    axs[i].imshow(image)
    axs[i].set_title('Tag: {}'.format(n), fontsize=10)
    axs[i].axis('off')
plt.show()
"""


f, axs =plt.subplots(1, 2,figsize=(20,20) )

for i in range(2):
    n=random.randint(0,9)
    image=x_train[y_train == n][0]         #selecting the images where y_label is same as int n
    axs[i].imshow(image,cmap ='gray')
    axs[i].set_title('Label: {}'.format(n), fontsize=20)
plt.show()

# Expand dimension of input dataset as keras takes 3 dimentsions always.
# This step is not required for color images
x_train, x_test = np.expand_dims(x_train,axis= -1), np.expand_dims(x_test,axis =-1)
print("x_train : ",x_train.shape)
print("x_test  : ",x_test.shape)

#%% Sequential API
"""
 It is ideal for building models where each layer has exactly one input tensor
 and one output tensor. Uusing the Sequential API is simple and straightforward,
 but is only appropriate for simpler, more straightforward tasks.

source: https://victorzhou.com/blog/keras-cnn-tutorial/
"""

def CNN_sequential_model():
    # CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    model = tf.keras.Sequential([
        tfl.ZeroPadding2D(padding =0,input_shape = (28,28,1)),
        tfl.Conv2D(filters=32,kernel_size=(3,3),strides=1,padding = 'valid',activation = 'relu'),
        tfl.MaxPool2D(),
        tfl.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding = 'valid',activation = 'relu'),
        tfl.MaxPool2D(),
        tfl.Flatten(),
        tfl.Dense(10,activation ='softmax')
        ])
    return model

#%% Tranining and evaluation
def train_evaluate_seq_API(seq_model,x_train,y_train,epoch,batchs_size,x_test,y_test):
    seq_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
    seq_model.summary()
    history = seq_model.fit(x_train, y_train,
                               batch_size=batchs_size, epochs= epoch,
                               verbose = 1,
                               validation_data = (x_test, y_test))

#%%Visualization of results
def visualization_Seq_CNN(seq_model,x_test,y_test):
    y_predictions =seq_model.predict(x_test)    #This gives the probability of each number
    y_pred=np.argmax(y_predictions,axis =1) #select the indices of prediction with highest probability
    x_test_2 = np.squeeze(x_test,axis =-1)    #restore the image dimension
    for i in range(5):
        n=random.randint(0,len(x_test_2))
        test_image=x_test_2[n]
        y_true = y_test[n]
        y_pred_val = y_pred[n]
        plt.title("Pred: {}, True: {}".format(y_pred_val,y_true))
        plt.imshow(test_image,cmap ='gray')
        plt.show() 
#%% save the model if evaluation results are acceptable
# model.save_weights('CNN_sequential_model.h5')
#model/load_weights('CNN_sequential_model.h5')


#%% Functional API
"""
The Functional API can handle models with non-linear topology, shared layers,
as well as layers with multiple inputs or outputs. Imagine that, where the
Sequential API requires the model to move in a linear fashion through its layers,
the Functional API allows much more flexibility. Where Sequential is a straight line,
a Functional model is a graph, where the nodes of the layers can connect in many more
ways than one.
"""
def CNN_functional_model(input_shape):
    # CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    inputs = keras.layers.Input(shape=input_shape)
    #Conv2d layer: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
    conv = keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides = (1,1),padding ='valid',activation = tf.nn.relu) (inputs)
    max_pool = keras.layers.MaxPool2D(pool_size =(2,2),strides = (2,2))(conv)
    conv = keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides = (1,1),padding ='valid',activation = tf.nn.relu) (max_pool)
    max_pool = keras.layers.MaxPool2D(pool_size =(2,2),strides = (2,2))(conv)
    flatten = keras.layers.Flatten()(max_pool)
    outputs = keras.layers.Dense(10,activation =tf.nn.softmax)(flatten)
    model = keras.models.Model(inputs,outputs)
    return model

#%% Train  and evaluate  functional API model
def train_evaluate_funct_API(funct_mdoel,x_train,y_train,epoch,batchs_size,x_test,y_test):
    #Compile model
    funct_mdoel.compile(optimizer='adam',
                      loss= 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    funct_mdoel.summary()
    # Training and evaluation of dataset
    history = funct_mdoel.fit(x_train, y_train,
                               batch_size=batchs_size, epochs=epoch,
                               verbose = 1,
                               validation_data = (x_test, y_test))

#%% visualization
def visualization_funct_CNN(funct_mdoel,x_test,y_test):
    y_predictions =funct_mdoel.predict(x_test)    #This gives the probability of each number
    y_pred=np.argmax(y_predictions,axis =1) #select the indices of prediction with highest probability
    x_test_3 = np.squeeze(x_test,axis =-1)    #restore the image dimension
    for i in range(5):
        n=random.randint(0,len(x_test_3))
        test_image=x_test_3[n]
        y_true = y_test[n]
        y_pred_val = y_pred[n]
        plt.title("Pred: {}, True: {}".format(y_pred_val,y_true))
        plt.imshow(test_image,cmap ='gray')
        plt.show() 

#%%
def main():
    option = 1 #1 =Functional API, 2= sequnetial API
    start= time.time()
    if option == 1:
        print('executing sequential model')
        seq_model = CNN_sequential_model()
        train_evaluate_seq_API(seq_model,x_train,y_train,10,512,x_test,y_test)
        visualization_Seq_CNN(seq_model,x_test,y_test)
    else: 
        print('executing functional model')
        funct_mdoel = CNN_functional_model(input_shape=(28,28,1))
        train_evaluate_funct_API(funct_mdoel,x_train,y_train,10,512,x_test,y_test)
        visualization_funct_CNN(funct_mdoel,x_test,y_test)
    end= time.time()
    print('total executaion time:', end-start)
    
#%%        
if __name__ == "__main__":
    main()