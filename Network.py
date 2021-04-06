from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
import random
import numpy as np
import math
import tensorflow as tf
from keras.utils import to_categorical
import DenseLayer

def Sigmoid(x):
    return tf.math.sigmoid(x)

class Network:
    def __init__(self):
       self.Layers = []
       self.Weights = []
       self.Biases = []

    def AddLayer(self, Layer):
        self.Layers.append(Layer)
        self.Weights.append(Layer.Weights)
        self.Biases.append(Layer.Biases)

    def FeedForward(self, x):
        for Layer in self.Layers:
            x = Layer.Mul(x)
        return x
 
    def Train(self, Data_Train, Data_Label, epochs, learning_rate= 0.01 ,batch_size = 100):
        Indexes = list( range(len(Data_Train)) )

        for epoch in range(epochs):
            print("Current epoch {0} -------------------------".format(epoch+1))
            random.shuffle(Indexes)      
            MiniBatches = [Indexes[k*batch_size : (k+1)*batch_size] for k in range(0, int (len(Data_Train)/batch_size) ) ] 
            for MiniBatch in MiniBatches:
                labels = np.reshape(Data_Label[MiniBatch],(len(MiniBatch), Data_Label[0].shape[0],1 ))
                gradient = self.TrainOnBatch(tf.Variable(Data_Train[MiniBatch]),tf.Variable(labels) )

                lol =np.array( gradient[0][2] )
                for index, (Weight ,Bias)  in enumerate(zip(self.Weights,self.Biases)):
                    Weight.assign_sub( (learning_rate) * gradient[0][index] )
                    Bias.assign_sub( (learning_rate) * gradient[1][index] )   
   
    def TrainOnBatch(self, Data_Train, DataLabel):    
        with tf.GradientTape() as tape:
            X = Data_Train
            X = self.FeedForward(X)
            loss = tf.keras.losses.MSE(DataLabel, X)
        return tape.gradient(loss,[self.Weights, self.Biases ])


    def Eval(self, Data, Labels):
        out =[]
        Data_Length = len(Labels)
        out = self.FeedForward(Data)
        out = np.array(out)
        out = np.reshape(out,( out.shape[0],out.shape[1] ))
        acc = 0
        for y_pred,y_real in zip(out, Labels):
            if np.argmax(y_pred) == np.argmax(y_real):
                acc+=1
        return float(acc/Data_Length)*100
        

Siec = Network()
Siec.AddLayer(DenseLayer.Dense(28*28, 64, Sigmoid) )
Siec.AddLayer(DenseLayer.Dense(64, 32, Sigmoid) )
Siec.AddLayer(DenseLayer.Dense(32, 10, Sigmoid) )

train_images = train_images.reshape((60000, 28* 28,1))
train_images = train_images.astype('float32') / 255 
train_labels = to_categorical(train_labels)


Siec.Train(train_images,train_labels, 15,batch_size=150)
#TrainImages = np.concatenate([train_images,train_images],axis=3)

test_images = test_images.reshape((10000, 28* 28,1))
test_images = test_images.astype('float32') / 255 
test_labels = to_categorical(test_labels)

print("Acc: " +  str (Siec.Eval(test_images,test_labels)) )