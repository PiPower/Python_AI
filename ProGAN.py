import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dense, LeakyReLU,\
    BatchNormalization, AveragePooling2D, Reshape, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
import numpy as np


class ConvLayerUp(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(ConvLayerUp, self).__init__(**kwargs)
        self.conv1 = Conv2D(out_channels, (3, 3), padding='same')
        self.conv2 = Conv2D(out_channels, (3, 3), padding='same')
        self.leakyRelu = LeakyReLU()
        self.up_sample = UpSampling2D()
        self.batchNorm = BatchNormalization()

    def call(self, x):
        x = self.up_sample(x)

        x = self.conv1(x)
        x = self.leakyRelu(x)
        x = self.batchNorm(x)

        x = self.conv2(x)
        x = self.leakyRelu(x)
        x = self.batchNorm(x)

        return x


class ConvLayerDown(tf.keras.layers.Layer):
    def __init__(self, out_channels, **kwargs):
        super(ConvLayerDown, self).__init__(**kwargs)
        self.conv1 = Conv2D(out_channels, (3, 3), padding='same')
        self.conv2 = Conv2D(out_channels, (3, 3), padding='same')
        self.leakyRelu = LeakyReLU()
        self.pool = AveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.leakyRelu(x)

        x = self.pool(x)

        return x


class Generator(tf.keras.models.Model):
    def __init__(self, input_shape, out_channels, img_channels, scale=1, alpha=0.5, layerFilterCount=None, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.sale = scale
        self.alpha = alpha
        self.max_layer = None
        self.img_channels = img_channels
        self.entry_shape = input_shape
        self.out_channels = out_channels
        self.to_RGB = [Conv2D(self.img_channels, (1, 1), activation='tanh', padding="same") for _ in range(scale)]

        self.call_stack = [Sequential(
            [
                BatchNormalization(),
                Dense(out_channels * input_shape[0] * input_shape[1]),
                LeakyReLU(),
                Reshape((input_shape[0], input_shape[1], out_channels)),
                BatchNormalization(),
                Conv2D(out_channels, (3, 3), padding='same'),
                LeakyReLU(),
                BatchNormalization()
            ])
        ]
        if layerFilterCount is not None:
            assert len(layerFilterCount) == scale - 1
            for outChannel in layerFilterCount:
                self.call_stack.append(ConvLayerUp(outChannel))
        else:
            for i in range(scale - 1):
                out_channels = out_channels / 2
                self.call_stack.append(ConvLayerUp(out_channels))
        '''
        for  i, conv in enumerate( self.to_RGB ):
            dim = [(i+1)* dim for dim in self.entry_shape] + [int( self.out_channels/(2**i) )]
            conv.build(dim)'''

    def call(self, x):
        y = None
        for i in range(self.max_layer):
            x = self.call_stack[i](x)
            if i == self.max_layer - 2:
                y = x
                y = self.to_RGB[self.max_layer - 2](y)
                y = UpSampling2D()(y)

        x = self.to_RGB[self.max_layer - 1](x)

        if y is not None:
            return self.alpha * x + (1 - self.alpha) * y
        else:
            return x

    def SetMaxLayer(self, ind):
        self.max_layer = ind


class Discriminator(tf.keras.models.Model):
    def __init__(self, input_shape, out_channels, scale=1, alpha=0.5, layerFilterCount=None, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.scale = scale
        self.entry_shape = input_shape
        self.out_channels = out_channels
        self.alpha = alpha
        self.start_index = None
        self.from_RGB = []
        self.call_stack = [Sequential(
            [
                Conv2D(out_channels, (3, 3), padding='same'),
                LeakyReLU(),
                Flatten(),
                Dense(out_channels),
                LeakyReLU(),
                Dense(1, activation='sigmoid'),
            ])
        ]

        if layerFilterCount is not None:
            for i in range(scale - 1):
                self.call_stack.append(ConvLayerDown(layerFilterCount[i]))
                self.from_RGB.append(Conv2D(layerFilterCount[i], (1, 1), padding="same"))
            self.from_RGB.append(Conv2D(layerFilterCount[scale - 1], (1, 1), padding="same"))
        else:
            self.from_RGB.append(Conv2D(out_channels / 2, (1, 1), padding="same"))
            for i in range(scale - 1):
                out_channels = out_channels / 2

                self.call_stack.append(ConvLayerDown(out_channels))
                self.from_RGB.append(Conv2D(out_channels / 2, (1, 1), padding="same"))
        '''
        for i, conv in enumerate(self.from_RGB):
            dim = [ 2**(i) * dim for dim in self.entry_shape] +  [3] #[int(self.out_channels / (2 ** (i+1) ) )  if i != len(self.from_RGB)-1 else 3 ]
            conv.build(dim)
        '''

        self.call_stack.reverse()
        self.from_RGB.reverse()

    def SetStartIndex(self, ind):
        self.start_index = ind

    def call(self, input):
        faded = False
        called = False

        x = self.from_RGB[self.start_index](input)

        for i in range(self.start_index, self.scale):
            if not faded and self.scale - i > 1:
                faded = True
                y = AveragePooling2D()(input)
                y = self.from_RGB[i+1](y)
                y = LeakyReLU()(y)

            x = self.call_stack[i](x)

            if not called and faded:
                called = True
                x = self.alpha * x + (1 - self.alpha) * y

        return x


class ProGAN():
    def __init__(self, latent_dim, entry_dim, img_channels, out_channels=128, scale=1,
                 GenAlpha=0.5, DiscAlpha=0.5, GenFilterList=None, DiscFilterList=None):
        self.latent_dim = latent_dim
        self.scale = scale
        self.generator = Generator(entry_dim, out_channels, img_channels, scale, GenAlpha, GenFilterList)
        self.discriminator = Discriminator(entry_dim, out_channels, scale, DiscAlpha, DiscFilterList)
        # self.discriminator.compile(optimizer=Adam(0.00025), loss='binary_crossentropy', metrics=['acc'] )

        self.model = Sequential([self.generator, self.discriminator])
        # self.model.compile(optimizer=Adam(0.00025), loss='binary_crossentropy', metrics=['acc'])

        self.model_optimizer = Adam(0.0001, 0, 0.999, 1e-8)
        self.discriminator_optimizer = Adam(0.00001, 0, 0.999, 1e-8)
        self.LossFunc = tf.keras.losses.BinaryCrossentropy()
        self.metric = tf.keras.metrics.Accuracy()
        # self._buildVariables()

    def _buildVariables(self):
        for i in range(self.scale - 1):
            self.discriminator.SetStartIndex(self.scale - (i + 1))
            self.generator.SetMaxLayer(i + 1)

            noise = tf.random.normal(shape=(3, self.latent_dim))
            fake_img = self.generator(noise)
            output = self.discriminator(fake_img)

    def train_on_batch(self, input_img, max_layer):
        self.discriminator.SetStartIndex(self.scale - max_layer)
        self.generator.SetMaxLayer(max_layer)
        
        noise = tf.random.normal(shape=(input_img.shape[0], self.latent_dim))
        fakeImgs = self.generator(noise)
        X = tf.concat([input_img, fakeImgs], axis=0)
        Y = tf.concat([tf.ones(shape=(input_img.shape[0], 1)) * 0.9, tf.zeros(shape=(input_img.shape[0], 1))], axis=0)

        self.discriminator.trainable = True
        with tf.GradientTape() as tape:
            pred = self.discriminator(X)
            dloss = self.LossFunc(Y, pred)
            gradients = tape.gradient(dloss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))
        self.metric.update_state(Y, pred)
        # disc_loss = self.discriminator.train_on_batch(X, Y)

        self.discriminator.trainable = False
        gen_input = tf.random.normal(shape=(input_img.shape[0], self.latent_dim))
        gen_label = tf.ones(shape=(input_img.shape[0], 1))
        with tf.GradientTape() as tape:
            IMGS = self.generator(gen_input)
            scores = self.discriminator(IMGS)
            mloss = self.LossFunc(gen_label, scores)
            gradients = tape.gradient(mloss, self.generator.trainable_variables)
            self.model_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        # total_loss = self.model.train_on_batch(gen_input , gen_label)

        print("Disc loss {0} Model loss {1} Disc accuracy: {2}".format(dloss, mloss, self.metric.result().numpy()))
        self.metric.reset_states()

