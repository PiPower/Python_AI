import  tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import  Adam
import numpy as np


class WaveNet():
    def __init__(self,timeSteps, dilatationSize, channels=16, kernelSize = 2, repetition = 2):
        self.timeStep = timeSteps
        self.DilatationFactor = [i**2 for i in range(1,dilatationSize+1)] * repetition
        self.channels = channels
        self.kernelSize = kernelSize
        skips = []

        z = Input( shape=(self.timeStep,1))
        x = Conv1D(channels, kernelSize, padding='causal', activation='relu', kernel_regularizer= L2(0.05) )(z)

        for dilatation in  self.DilatationFactor:

            x_f = Conv1D(channels, kernelSize, padding='causal',dilation_rate=dilatation, kernel_regularizer= L2(0.05) )(x)
            x_g = Conv1D(channels, kernelSize, padding='causal',dilation_rate=dilatation, kernel_regularizer= L2(0.05) )(x)

            g = tf.tanh(x_f) * tf.sigmoid(x_g)
            g = Conv1D(128, 1, kernel_regularizer= L2(0.05) )(g)  
            skips.append(g)
            x = x + g

        a = Add()(skips)
        a = ReLU()(a)
        a = Conv1D(64, 1, activation='relu', kernel_regularizer= L2(0.05) )(a)
        a = Conv1D(1, 1, activation='relu', kernel_regularizer= L2(0.05) )(a)

        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        a = Lambda(slice, arguments={'seq_length': timeSteps })(a)
        self.model = Model(z, a)
        self.model.compile(Adam(0.00025), loss='mean_absolute_error', metrics= ['acc','MSE'])

    def train(self, dataset, epochs, dataset_val):
        self.model.fit(x = dataset, epochs = epochs, validation_data = dataset_val)
    def predict(self, data):
        return self.model.predict(data)
