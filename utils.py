import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D,LayerNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.constraints import Constraint
import numpy as np
import os
import matplotlib.pyplot as plt

def Procces(x, y = None):
    x = x / 127.5 - 1
    # x, y = tf.split(x, 2, axis=-2)
    return x, y


def decode_img(img,x_res, y_res):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [x_res, y_res])


def process_path(file_path, y=None, x_res = 80, y_res = 80):
    img = tf.io.read_file(file_path)
    img = decode_img(img, x_res, y_res )
    if y is not None: return img, y
    return img


def build_convolutional_generator(laten_dim, img_cols, channels):
    z = Input(shape=(laten_dim,))
    x = Dense(128 * int(img_cols / 4) * int(img_cols / 4), activation="relu", input_dim=laten_dim,
              kernel_regularizer=l2(0.15))(z)
    x = Reshape((int(img_cols / 4), int(img_cols / 4), 128))(x)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),
           kernel_initializer=Orthogonal())(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)

    x = UpSampling2D()(x)
    a = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),
               kernel_initializer=Orthogonal())(x)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),
               kernel_initializer=Orthogonal())(a)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = a + x

    x = UpSampling2D()(x)
    a = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),
               kernel_initializer=Orthogonal())(x)
    x = Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),
               kernel_initializer=Orthogonal())(a)
    x = BatchNormalization(momentum=0.8)(x)
    x = ReLU()(x)
    x = a + x

    x = Conv2D(channels, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15), activation='tanh',
               kernel_initializer=Orthogonal())(x)
    return Model(z, x)


def build_discriminator(img_shape, kernel_constraint = None, bias_constraint = None, last_activation = 'sigmoid'):
    z = Input(shape=img_shape)

    a = BatchNormalization(momentum=0.8)(z)
    a = Conv2D(45, kernel_size=(1, 1), kernel_regularizer=l2(0.15), kernel_initializer=Orthogonal(),
               kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(a)
    a = LeakyReLU()(a)
    a = AveragePooling2D()(a)

    x = BatchNormalization(momentum=0.8)(z)
    x = LeakyReLU()(x)
    x = Conv2D(45, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),kernel_initializer=Orthogonal(),
               kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(x)
    x = LeakyReLU()(x)
    x = Conv2D(45, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),kernel_initializer=Orthogonal(),
               kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(x)
    x = AveragePooling2D()(x)

    x = a + x

    a = BatchNormalization(momentum=0.8)(x)
    a = Conv2D(80, kernel_size=(1, 1), kernel_regularizer=l2(0.15), kernel_initializer=Orthogonal(),
               kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(a)
    a = LeakyReLU()(a)
    a = AveragePooling2D()(a)

    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU()(x)
    x = Conv2D(80, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),kernel_initializer=Orthogonal(),
               kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(x)
    x = LeakyReLU()(x)
    x = Conv2D(80, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(0.15),kernel_initializer=Orthogonal(),
               kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(x)
    x = AveragePooling2D()(x)

    x = a + x

    x = Flatten()(x)

    x = Dense(1, activation=last_activation, kernel_regularizer=l2(0.15),  kernel_constraint = kernel_constraint, bias_constraint = bias_constraint )(x)

    return Model(z, x)

class CreateSample(tf.keras.callbacks.Callback):

    def __init__(self, model, path, image_name_convetion):
        super(CreateSample, self).__init__()
        self.model = model
        self.path = path
        self.image_name_convetion = image_name_convetion

    def on_epoch_end(self, epoch, logs=None):
        r, c = 4, 4
        noise = np.random.normal(0, 1, size=[r * c, self.model.latent_dim])
        gen_imgs = self.model.generator.predict(noise, verbose=0)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(30, 30))
        cnt = 0
        for i in range(4):
            for j in range(4):
                img = gen_imgs[cnt]
                # img = np.concatenate([img,img,img], axis=-1)
                if self.model.channels == 1:
                    axs[i, j].imshow(img, cmap='gray')
                else:
                    axs[i, j].imshow(img)
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(self.path, self.image_name_convetion) + "_%d.png" % epoch)
        plt.close()

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return tf.clip_by_value(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}