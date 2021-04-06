import tensorflow as tf

class Dense:
    def __init__(self, Input_Dim, Output_Dim, Activation):
        init = tf.initializers.glorot_normal()
        self.Weights = tf.Variable(init(shape=(Output_Dim, Input_Dim)) )
        self.Biases = tf.Variable( init(shape=(Output_Dim, 1)) )
        self.Input_Dim = Input_Dim
        self.Output_Dim = Output_Dim
        self.Activation = Activation
    
    def Mul(self, x):
        return self.Activation(tf.linalg.matmul(self.Weights,x) + self.Biases )

