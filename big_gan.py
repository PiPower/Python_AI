import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D,LayerNormalization
from tensorflow.keras.layers import LeakyReLU, ReLU, Embedding, MaxPool2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.regularizers import OrthogonalRegularizer
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
import os
import matplotlib.pyplot as plt
import numpy as np

class CreateSampleBigGan(tf.keras.callbacks.Callback):

    def __init__(self, model, path, image_name_convetion, save_freq = 4):
        super(CreateSampleBigGan, self).__init__()
        self.model = model
        self.path = path
        self.save_freq = save_freq
        self.image_name_convetion = image_name_convetion

    def on_epoch_end(self, epoch, logs=None):
        if epoch% self.save_freq == 0:
            r, c = 4, 4
            gen_imgs = self.model.generate(r * c)[0]
            gen_imgs = 0.5 * gen_imgs + 0.5

            gen_imgs = tf.clip_by_value(gen_imgs, 0, 1)
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
@tf.function
def discriminator_hinge_loss(real, fake):
    loss_real = tf.math.reduce_mean(tf.nn.relu(1. - real))
    loss_fake = tf.math.reduce_mean(tf.nn.relu(1. + fake))
    return loss_real, loss_fake
@tf.function
def generator_hinge_loss(fake):
    loss = -tf.math.reduce_mean(fake)
    return loss

def non_local_block(filters, height, width, spectral_norm = True):
    assert filters%2 ==0
    x = Input( shape = (height, width, filters) )
    if spectral_norm:
        g = SpectralNormalization(  Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(), kernel_regularizer = OrthogonalRegularizer() ) )(x)
        theta = SpectralNormalization(  Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(), kernel_regularizer = OrthogonalRegularizer() ) )(x)
        phi = SpectralNormalization( Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )(x)
    else:
        g = Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) (x)
        theta = Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() )(x)
        phi = Conv2D(filters/2, (1,1), padding = 'same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() )(x)

    phi = MaxPool2D()(phi)
    g = MaxPool2D()(g)

    theta = tf.reshape(theta, (-1,height*width, filters // 2))
    phi = tf.reshape(phi, (-1, filters // 2, height * width//4,))
    g = tf.reshape(g, (-1, height*width//4, filters//2))

    y = tf.nn.softmax( tf.matmul(theta, phi), 1 )
    z = tf.matmul(y, g)
    z = tf.reshape(z, (-1, height, width, filters//2))
    if spectral_norm:
        z =  SpectralNormalization( Conv2D(filters, (1,1),
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )(z)
    else:
        z = Conv2D(filters, (1, 1), kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() )(z)
    z = z + x

    return tf.keras.models.Model(x,z)

class ConditionalBatchNormalization(tf.keras.layers.Layer):
    index = 0
    def __init__(self, class_count, filter_size, epsilon = 0.00001, momentum=0.99, **kwargs):
        super(ConditionalBatchNormalization, self).__init__(**kwargs)

        self.class_count = class_count
        self.gamma =  SpectralNormalization( Dense(filter_size,
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
                                             #name="cond_bn_{0}_SN:kernel1".format(ConditionalBatchNormalization.index) )
        self.beta =  SpectralNormalization( Dense(filter_size,
                                           kernel_initializer = Orthogonal(), kernel_regularizer = OrthogonalRegularizer() ))
                                            # name="cond_bn_{0}_SN:kernel2".format(ConditionalBatchNormalization.index) )
        self.epsilon =epsilon
        self.filters = int(filter_size)
        self.moving_mean = tf.Variable([0.0]*self.filters, dtype = tf.float32, name = "moving_mean:{0}".format(self.index), trainable = False)
        self.moving_var = tf.Variable([1.0]*self.filters, dtype = tf.float32, name = "moving_var:{0}".format(self.index), trainable = False)
        self.momentum = momentum
        ConditionalBatchNormalization.index = ConditionalBatchNormalization.index+1

    @tf.function
    def call(self, x, z, training):
        beta = self.beta(z,  training=training)
        gamma = self.gamma(z,  training=training)
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.filters])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.filters])

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.moving_mean.assign( self.moving_mean * self.momentum + batch_mean * (1-self.momentum))
            self.moving_var.assign(self.moving_var * self.momentum + batch_var * (1 - self.momentum) )
            return tf.nn.batch_normalization(x, batch_mean, batch_var,beta,gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.moving_mean, self.moving_var, beta, gamma, self.epsilon)


class ResBlock_Up(tf.keras.layers.Layer):
    def __init__(self, filters, class_count, first = False, **kwargs):
        super(ResBlock_Up, self).__init__(**kwargs)
        if first:
            self.linear_1 = SpectralNormalization( Dense(filters,
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
            self.batch_norm_1 = ConditionalBatchNormalization(class_count, filters)
        else:
            self.linear_1 =  SpectralNormalization( Dense(2*filters,
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
            self.batch_norm_1 = ConditionalBatchNormalization(class_count, 2*filters)
        self.linear_2 =  SpectralNormalization( Dense(filters,
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
        self.batch_norm_2 = ConditionalBatchNormalization(class_count, filters)
        self.relu_1 = ReLU()
        self.relu_2 = ReLU()
        self.up_sample_1 = UpSampling2D()
        self.up_sample_2 = UpSampling2D()
        self.conv_1_3x3 = SpectralNormalization( Conv2D(filters, (3,3), padding = 'same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
        self.conv_2_3x3 = SpectralNormalization(  Conv2D(filters, (3, 3), padding='same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
        self.conv_3_1x1 = SpectralNormalization(Conv2D(filters, (3, 3), padding='same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )

    @tf.function
    def call(self, x, class_x, training):
        y = self.up_sample_1(x)
        y = self.conv_3_1x1( y,  training=training)

        z = self.batch_norm_1(x, class_x, training)
        z = self.relu_1(z)
        z = self.up_sample_2(z)
        z = self.conv_1_3x3(z,  training=training)
        z = self.batch_norm_2(z,class_x,training)
        z = self.relu_2(z)
        z = self.conv_2_3x3(z,  training=training)
        return y+z

class ResBlock_Down(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResBlock_Down, self).__init__(**kwargs)
        self.conv_1_3x3 = SpectralNormalization( Conv2D(filters, (3, 3), padding='same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
        self.conv_2_3x3 = SpectralNormalization( Conv2D(filters, (3, 3), padding='same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ) )
        self.conv_3_1x1 = SpectralNormalization( Conv2D(filters, (3, 3), padding='same',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() ))
        self.relu_1 = ReLU()
        self.relu_2 = ReLU()
        self.average_pooling_1 = AveragePooling2D()
        self.average_pooling_2 = AveragePooling2D()

    @tf.function
    def call(self, x, training):
        y = self.conv_3_1x1(x,  training=training)
        y = self.average_pooling_1(y)

        z = self.relu_1(x)
        z = self.conv_1_3x3(z,  training=training)
        z = self.relu_2(z)
        z = self.conv_2_3x3(z,  training=training)
        z = self.average_pooling_2(z)
        return z+y



class BigGan_Generator(tf.keras.models.Model):
    def __init__(self, res_block_count, latent_dim, image_dim, class_count, hidden_channels = 10, non_local_index = None, embedd_dim = 128, **kwargs):
        super(BigGan_Generator,  self).__init__(**kwargs)
        assert  latent_dim%(res_block_count+1) == 0
        self.height = image_dim[0]
        self.width = image_dim[1]
        self.channels = image_dim[2]
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.class_embedding = Embedding(class_count, embedd_dim)
        self.res_block_count = res_block_count
        self.layer_multiplayer = (2**(res_block_count-1))*self.hidden_channels
        self.blocks = []
        for i in range(res_block_count):
            self.blocks.append(ResBlock_Up( self.layer_multiplayer/(2**i), embedd_dim, i == 0))
            if non_local_index is not None and non_local_index == i:
                self.blocks.append(non_local_block(  self.layer_multiplayer//(2**i), self.height * (2**(i+1)),self.width * (2**(i+1)) ) )
        self.MLP = Dense(self.height * self.width * self.layer_multiplayer, kernel_regularizer = OrthogonalRegularizer() )
        self.batch_norm = BatchNormalization()
        self.relu= ReLU()
        self.out_conv_3x3 = Conv2D( self.channels, (3, 3), padding='same',activation = 'tanh',
                                           kernel_initializer = Orthogonal(),  kernel_regularizer = OrthogonalRegularizer() )
        #self.non_local = non_local_block(  self.hidden_channels,self.height * (2**(res_block_count)),self.width * (2**(res_block_count))   )

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = inputs[0]
        c_emb = self.class_embedding(inputs[1])
        c_emb = tf.squeeze(c_emb)
        z_split = tf.split(x, self.res_block_count+1, axis=-1)

        img = self.MLP(z_split[0],  training=training)
        img = tf.reshape( img, (-1,self.height,self.width,self.layer_multiplayer))

        z_index = 1
        for block in self.blocks:
            if isinstance(block, ResBlock_Up):
                concat_tesnor = tf.concat([z_split[z_index], c_emb], axis=-1)
                img = block(img, concat_tesnor, training )
                z_index = z_index + 1
            else:
                img = block(img,training)

        img = self.batch_norm(img,training)
        img = self.relu(img)
        img = self.out_conv_3x3(img,  training=training)
        return img

class BigGan_Discriminator(tf.keras.models.Model):
    def __init__(self, block_count,class_count, height, width, hidden_channels = 10, last_activation = None,non_local_index = None, **kwargs):
        super(BigGan_Discriminator, self).__init__(**kwargs)
        self.blocks = []
        for i in range(block_count):
            self.blocks.append(ResBlock_Down((2**(i+1))*hidden_channels ) )
            if non_local_index is not None and non_local_index == i:
                self.blocks.append(  non_local_block((2**(i+1))*hidden_channels, height//(2**(i+1)),  width//(2**(i+1)), True)   )

        self.blocks.append( ResBlock_Down((2**(block_count))*hidden_channels ))
        self.linear = Dense(1, activation = last_activation, kernel_regularizer = OrthogonalRegularizer() )
        self.relu = ReLU()
        self.class_embedding = Embedding(class_count, (2**(block_count))*hidden_channels )
        #self.non_local = non_local_block(hidden_channels*2 ,14,14, True)

    @tf.function
    def call(self, inputs,  training=None, mask=None):
        img = inputs[0]
        y = inputs[1]

        for i, block in enumerate(self.blocks):
            img = block(img, training)

        img = self.relu(img)
        pooled = tf.math.reduce_sum(img, axis=[1,2])

        emb = tf.squeeze( self.class_embedding(y), axis=1)
        class_regularization = tf.reduce_sum(emb * pooled, axis=1)
        class_regularization = tf.reshape(class_regularization, (-1,1))

        out = self.linear(pooled,  training=training) + class_regularization
        return out

class BigGan(tf.keras.models.Model):
    def __init__(self,block_count,  latent_dim, ground_height, ground_width,image_channels, class_count,
                 hidden_channels = 10,embedd_dim = 128, last_activation = None,n_crit = 2, D_accumulate = 1,
                non_local_gen = None,non_local_dis = None, **kwargs):
        super(BigGan, self).__init__(**kwargs)
        assert D_accumulate >= 1
        self.class_count = class_count
        self.n_crit_max = n_crit
        self.curr_crit = 0
        self.latent_dim = latent_dim
        self.channels = image_channels
        self.D_accumulate = D_accumulate
        self.generator = BigGan_Generator(block_count, latent_dim,(ground_height, ground_width, image_channels),
                                          class_count, hidden_channels,non_local_gen, embedd_dim)
        self.discriminator = BigGan_Discriminator(block_count, class_count,ground_height*(2**block_count),
        ground_width*(2**block_count), hidden_channels, last_activation, non_local_dis)
        self.accumulated = []
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    def generate(self, size, y = None, training = False):
        image_noise = tf.random.normal((size, self.latent_dim))
        class_label = y
        if y is None:
            class_label = tf.random.uniform((size, 1), 0, self.class_count, dtype=tf.int32)
        return self.generator((image_noise,class_label), training = training), class_label

    def compile(self, dis_loss =discriminator_hinge_loss, discriminator_optimizer = Adam(0.0002, 0),
                gen_loss = generator_hinge_loss,  generator_optimizer = Adam(0.00002, 0)):
        super(BigGan, self).compile(run_eagerly = True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)
        self.dis_loss = dis_loss
        self.generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
        self.gen_loss = generator_hinge_loss

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        dot_index = filepath.rfind(".")
        filepath_generator = filepath[:dot_index] + "_generator" + filepath[dot_index:]
        self.generator.save_weights(filepath_generator)

        filepath_discriminator = filepath[:dot_index] + "_discriminator" + filepath[dot_index:]
        self.discriminator.save_weights(filepath_discriminator)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        dot_index = filepath.rfind(".")
        filepath_generator = filepath[:dot_index] + "_generator" + filepath[dot_index:]
        self.generator.load_weights(filepath_generator)

        filepath_discriminator = filepath[:dot_index] + "_discriminator" + filepath[dot_index:]
        self.discriminator.load_weights(filepath_discriminator)

    def summary(self):
        print("generator summary")
        self.generator.summary()
        print("discriminator summary")
        self.discriminator.summary()

    def call(self, inputs, training=None, mask=None):
        z = self.generate(inputs)
        return self.discriminator(z)
    @tf.function
    def train_discriminator(self, real_images,class_labels,  fake_images, class_labels_fake):
        pred_real = self.discriminator((real_images, class_labels), training=True)
        pred_fake = self.discriminator((fake_images, class_labels_fake), training=True)

        loss_real, loss_fake = self.dis_loss(pred_real, pred_fake)
        dloss = (loss_fake + loss_real) / float(self.D_accumulate)
        return dloss

    def train_generator(self, real_images):
        images, class_labels = self.generate(tf.shape(real_images)[0], training=True)
        pred_fake = self.discriminator((images, class_labels))
        gloss = self.gen_loss(pred_fake)
        return gloss

    def train_step(self, x):
        real_images = x[0]
        class_labels = x[1]
        # training loop for discriminator
        if self.curr_crit < (self.n_crit_max*self.D_accumulate):
            batch_size = tf.shape(real_images)[0]
            fake_images, class_labels_fake = self.generate(batch_size)

            with tf.GradientTape() as tape:
                dloss = self.train_discriminator(real_images, class_labels, fake_images, class_labels_fake)

            trainable_vars = self.discriminator.trainable_variables
            gradients = tape.gradient(dloss, trainable_vars)

            if len(self.accumulated) == self.D_accumulate - 1:
                for grad_list in  self.accumulated:
                    for i in range(len(self.accumulated)):
                        gradients[i] = gradients[i] + grad_list[i]

                self.accumulated = []
                self.discriminator.optimizer.apply_gradients(zip(gradients, trainable_vars))
            else: self.accumulated.append(gradients)

            self.d_loss_tracker.update_state(dloss)
            self.curr_crit += 1
            return {
                "d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result(),
            }
        #training loop for generator
        self.curr_crit = 0
        with tf.GradientTape() as tape:
            gloss = self.train_generator(real_images)

        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(gloss, trainable_vars)
        self.generator.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.g_loss_tracker.update_state(gloss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }


lol = SpectralNormalization(Dense(1))
data = np.asarray([  [20,304,10], [192,3902, 192]  ])
lol(data, True)
xd = lol.trainable_variables
xd = 's'