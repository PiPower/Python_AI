from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, Conv2D, UpSampling2D, AveragePooling2D
from utils import *


@tf.function
def pixel_norm(x, epsilon = 1e-8):
    out = x/tf.sqrt( tf.reduce_mean(x**2, axis=-1, keepdims=True) + epsilon )
    return out

class ProGanInitialzier(tf.keras.initializers.Initializer):
    def __init__(self, **kwargs):
        super(ProGanInitialzier, self).__init__(**kwargs)

    def _compute_fans(self, shape):
        """Computes the number of input and output units for a weight shape.
        Args:
          shape: Integer shape tuple or TF tensor shape.
        Returns:
          A tuple of integer scalars (fan_in, fan_out).
        """
        if len(shape) < 1:  # Just to avoid errors for constants.
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assuming convolution kernels (2D, 3D, or more).
            # kernel shape: (..., input_depth, depth)
            receptive_field_size = 1
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return int(fan_in), int(fan_out)

    def __call__(self, shape, dtype=None, **kwargs):
        w = tf.random.normal(shape)
        fan_in, _ = self._compute_fans(shape)
        w = w * tf.sqrt( 2.0/fan_in)
        return w



class AdaIn(tf.keras.layers.Layer):
    def __init__(self,filters,epsilon = 1e-8, **kwargs):
        super(AdaIn, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.filters = filters
        self.d_scale = Dense(filters, activation = LeakyReLU(0.2), kernel_initializer =ProGanInitialzier() )
        self.d_bias = Dense(filters, activation=LeakyReLU(0.2), kernel_initializer=ProGanInitialzier())

    @tf.function
    def call(self, x, w):
        x = x - tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + self.epsilon)

        y_scale = self.d_scale(w)
        y_scale = tf.reshape( y_scale, [-1,1,1,self.filters] )
        y_bias = self.d_bias(w)
        y_bias = tf.reshape( y_bias, [-1,1,1,self.filters] )

        x = y_scale*x + y_bias
        return x

class StyleGan_Up(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(StyleGan_Up, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters, (3, 3), activation=LeakyReLU(0.2), kernel_initializer=ProGanInitialzier(), padding='same')
        self.conv2 = Conv2D(filters, (3, 3), activation=LeakyReLU(0.2), kernel_initializer=ProGanInitialzier(), padding='same')
        self.up_sample = tf.keras.layers.UpSampling2D( interpolation = "bilinear")
        self.adain_1 = AdaIn(filters)
        self.adain_2 = AdaIn(filters)

    @tf.function
    def call(self,x, w):
        up_sample = self.up_sample(x)
        x = self.conv1(up_sample)
        noise_shape = (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1  )
        noise = tf.random.normal( (noise_shape)   )
        x = x + noise
        x = self.adain_1(x,w)
        x = self.conv2(x)
        x = self.adain_2(x,w)

        return x, up_sample

class Minibatch_StdDev(tf.keras.layers.Layer):
    def __init__(self, group_size, **kwargs):
        kwargs['trainable'] = False
        super(Minibatch_StdDev, self).__init__(**kwargs)
        self.group_size = tf.Variable(group_size, name="group_size", trainable= False)

    @tf.function
    def call(self,x):
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = tf.shape(x) # [NHWC]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)  # [GMHWC] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMHWC] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MHWC]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, tf.dtypes.float32)  # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])  # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=-1)  # [NHWC]  Append as new fmap.

class ConvBlock_DOWN(tf.keras.layers.Layer):
    def __init__(self, filters, double_filters = False, last = False, **kwags):
        super(ConvBlock_DOWN, self).__init__(**kwags)
        last_kernel_filter_size = (3,3)
        self.conv1 = Conv2D(filters, (3,3), activation = LeakyReLU(0.2), kernel_initializer =ProGanInitialzier(), padding = 'same' )

        padding = 'same'
        if last:
            last_kernel_filter_size = (4,4)
            padding = "valid"

        if double_filters:
            filters = filters * 2

        self.conv2 = Conv2D(filters,last_kernel_filter_size, activation=LeakyReLU(0.2), kernel_initializer=ProGanInitialzier(), padding=padding)

        self.downsample = AveragePooling2D()
        if last:
            self.downsample = Flatten()

    @tf.function
    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.downsample(x)
        return x

class To_RGB(tf.keras.layers.Layer):
    def __init__(self, channels,act = None,  **kwargs):
        super(To_RGB, self).__init__( **kwargs)
        self.channels = channels
        self.conv = Conv2D(channels, activation = act,kernel_size = 1,kernel_initializer=ProGanInitialzier(), padding = 'same')

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)

class StyleGan_Generator(tf.keras.models.Model):
    def __init__(self,hidden_dim, hidden_laten_dim, map_net_size = 8, start_height = 4, start_width = 4,
                 start_decreasing = 0, channels=3,  **kwargs):
        super(StyleGan_Generator, self).__init__(**kwargs)
        self.mapping_network = tf.keras.models.Sequential([
            Dense(hidden_laten_dim, activation = LeakyReLU(0.2), kernel_initializer = ProGanInitialzier()) for i in range(map_net_size)
        ])
        self.blocks = []
        self.channels = channels
        self.to_rgb_old = To_RGB(self.channels)
        self.to_rgb_real = To_RGB(self.channels)
        self.const = tf.Variable(tf.ones((1,start_height, start_width,hidden_dim)), name="Const", trainable=True)
        self.start_height = start_height
        self.start_width = start_width
        self.start_decreasing = start_decreasing
        self.hidden_dim = hidden_dim
        self.adain_1 = AdaIn(hidden_dim)
        self.adain_2 = AdaIn(hidden_dim)
        self.conv = Conv2D(hidden_dim, (3,3),activation = LeakyReLU(0.2) ,  kernel_initializer = ProGanInitialzier(),padding= "same")

        self.call = tf.function( self.call_prototype )

    def calc_hidden_dim(self, iteration ):
            if iteration > self.start_decreasing:
                return self.hidden_dim//( 2**(iteration - self.start_decreasing)  )
            else:
                return self.hidden_dim

    def add_layer(self):
        curr_hidden_dim = self.calc_hidden_dim(len(self.blocks))
        self.blocks.append(StyleGan_Up(curr_hidden_dim))

        self.to_rgb_old = To_RGB(self.channels)
        self.to_rgb_real = To_RGB(self.channels)
        """
        this is importan part if we want to squeeze maximu performance. tf.function compiles our function
        to graph but our function changes dynamicly with additional layer so we MUST recompile it every time 
        we add additional block.
        """
        self.call = tf.function(self.call_prototype)

    def call(self, x, alpha, training=None, mask=None):
        pass

    def call_prototype(self, x, alpha, training=None, mask=None):
        w = self.mapping_network(x)

        noise = tf.random.normal([tf.shape(x)[0],self.start_height, self.start_width, 1])
        img = noise + self.const
        img = self.adain_1(img,w)
        img = self.conv(img)
        img = self.adain_2(img, w)

        old_upsampled = img * 0
        for block in self.blocks:
            img, old_upsampled = block(img, w)

        image =  self.to_rgb_real(img)
        normalization = self.to_rgb_old(old_upsampled)

        return image * alpha + normalization * (1-alpha)


class StyleGan_Discriminator(tf.keras.models.Model):
    def __init__(self,hidden_dim, start_decreasing, group_size, **kwargs):
        super(StyleGan_Discriminator,self).__init__(**kwargs)
        self.steps = 1
        self.hidden_dim = hidden_dim
        self.start_decreasing = start_decreasing
        self.linear = Dense(1)
        self.from_rgb_old = lambda x:  0
        self.from_rgb_real = To_RGB(self.hidden_dim)
        self.downsample = AveragePooling2D()
        self.blocks = [ tf.keras.models.Sequential(
            [Minibatch_StdDev(group_size), ConvBlock_DOWN(self.hidden_dim, last = True)]
        )]
        self.call = tf.function(  self.call_prototype )

    def calc_hidden_dim(self, iteration):
        if iteration > self.start_decreasing:
            return self.hidden_dim // (2 ** (iteration - self.start_decreasing))
        else:
            return self.hidden_dim

    def add_layer(self):
        curr_hidden_dim = self.calc_hidden_dim(len(self.blocks))
        double_filters = False
        if len(self.blocks) > self.start_decreasing:
            double_filters = True

        self.blocks.append(ConvBlock_DOWN(curr_hidden_dim, double_filters))

        self.downsample = AveragePooling2D()
        self.from_rgb_old = To_RGB(curr_hidden_dim* ( 1 + int(double_filters)) )
        self.from_rgb_real = To_RGB(curr_hidden_dim)
        self.steps = self.steps + 1
        """
        this is importan part if we want to squeeze maximu performance. tf.function compiles our function
        to graph but our function changes dynamicly with additional layer so we MUST recompile it every time 
        we add additional block.
        """
        self.call = tf.function( self.call_prototype )

    def call(self, x, alpha, training=None, mask=None):
        pass

    def call_prototype(self, x, alpha, training=None, mask=None):
        img_downsampled = self.downsample(x)
        img_normalization = self.from_rgb_old(img_downsampled)

        img = self.from_rgb_real(x)
        img = self.blocks[-1](img)

        img = img * alpha + (1-alpha) * img_normalization

        for block in reversed(self.blocks[:len(self.blocks)-1]):
            img = block(img)

        return self.linear(img)


class StyleGan(tf.keras.models.Model):
    def __init__(self, hidden_dim, latent_dim, channels=3, blocks=4, start_dec=1,
                 start_height=4, start_width=4, n_critic = 1, gamma = 1,lambda_par = 10,
                 epsilon_drift = 0.001, group_size = 8,hidden_laten_dim = 256,map_net_size =8, **kwargs):
        super(StyleGan, self).__init__(**kwargs)
        assert blocks > start_dec
        self.n_critic = n_critic #tf.Variable(n_critic, name = "n_critic", trainable= False)
        self.gamma = gamma
        self.epsilon_drift = tf.Variable(epsilon_drift, trainable= False)
        self.lambda_par = tf.Variable(lambda_par, trainable= False, dtype=tf.float32)
        self.iteration =0# tf.Variable(0, name = "iter const", trainable= False)
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.latent_dim = latent_dim
        self.blocks = blocks
        self.start_dec = start_dec
        self.generator = StyleGan_Generator(hidden_dim, hidden_laten_dim,map_net_size, start_height, start_width, start_dec, channels )
        self.discriminator = StyleGan_Discriminator(hidden_dim, start_dec, group_size)
        self.alpha = tf.Variable(1.0, name="alpha", dtype=tf.float32, trainable= False)
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.train_discriminator = tf.function(self.train_discriminator_proto)
        self.train_generator = tf.function(self.train_generator_proto)

    def compile(self,discriminator_optimizer = Adam(0.00001, 0, 0.99), generator_optimizer = tf.keras.optimizers.Adam(0.00001, 0, 0.99) ):
        super(StyleGan, self).compile(run_eagerly= True)
        generator_optimizer = tfa.optimizers.MovingAverage(generator_optimizer)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer)
        self.generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)


    def train_discriminator(self, x):
        pass

    def train_discriminator_proto(self, x):
        with tf.GradientTape() as tape:
            z = tf.random.normal( (tf.shape(x)[0], self.latent_dim))
            epsilon = tf.random.uniform((tf.shape(x)[0],1,1,1), minval=0, maxval=1 )
            fake_img = self.generator(z, self.alpha)
            with tf.GradientTape() as tape_mixed:
                tape_mixed.watch(fake_img)
                mixed_image = epsilon * x + (1 - epsilon) * fake_img
                mixed_image_pred = self.discriminator(mixed_image, self.alpha, True)

            grad_penalty = tape_mixed.gradient(mixed_image_pred, mixed_image)
            grad_penalty_normalized =  tf.square( tf.math.l2_normalize(grad_penalty)-1 )
            grad_penalty_normalized = tf.reduce_mean(grad_penalty_normalized)
            dloss = tf.reduce_mean(self.discriminator(fake_img, self.alpha, True) ) - tf.reduce_mean(self.discriminator(x, self.alpha, True)) \
            + self.lambda_par * grad_penalty_normalized + self.epsilon_drift * tf.reduce_mean(  tf.square(self.discriminator(x, self.alpha, True)) )

        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(dloss, trainable_vars)
        self.discriminator.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return dloss

    def train_generator(self, x):
        pass

    def train_generator_proto(self, x):
        with tf.GradientTape() as tape:
            noise = tf.random.normal( (tf.shape(x)[0], self.latent_dim) )
            img = self.generator(noise, self.alpha)
            gloss = -tf.reduce_mean( self.discriminator(img, self.alpha) )
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(gloss, trainable_vars)
        self.generator.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return gloss

    def summary(self, *arg):
        self.generator.summary(*arg)
        self.discriminator.summary(*arg)

    def generate(self, count):
        z = tf.random.normal((count,self.latent_dim) )
        return self.generator(z, tf.Variable(1.0, dtype=tf.float32, trainable= False)), None

    def grow(self):
        self.generator.add_layer()
        self.discriminator.add_layer()
        self.train_discriminator = tf.function(self.train_discriminator_proto)
        self.train_generator = tf.function(self.train_generator_proto)

    def train_step(self, x):
        if self.iteration < self.n_critic:
            self.iteration = self.iteration + 1
            dloss = self.train_discriminator(x)

            self.d_loss_tracker.update_state(dloss)
            return {
                "d_loss": self.d_loss_tracker.result(),
                "g_loss": self.g_loss_tracker.result(),
            }

        self.iteration = 0
        gloss = self.train_generator(x)
        self.g_loss_tracker.update_state(gloss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }

    def save_weights(self, filepath,**kwargs):
        dot_index = filepath.rfind(".")
        filepath_generator = filepath[:dot_index] + "_generator" + filepath[dot_index:]
        self.generator.save_weights(filepath_generator, **kwargs)

        filepath_discriminator = filepath[:dot_index] + "_discriminator" + filepath[dot_index:]
        self.discriminator.save_weights(filepath_discriminator, **kwargs)


if __name__ == "__main__":
    progan = StyleGan(256, 120, start_dec=0, channels=1)
    progan.compile()
    noise = tf.random.normal( (64,4,4,1) )
    progan.train_step(noise)
    progan.train_step(noise)
    progan.grow()
    noise = tf.random.normal( (64,8,8,1) )
    progan.train_step(noise)
    progan.train_step(noise)
    progan.grow()
    noise = tf.random.normal( (64,16,16,1) )
    progan.train_step(noise)
    progan.train_step(noise)
    progan.grow()
    noise = tf.random.normal( (64,32,32,1) )
    progan.train_step(noise)
    progan.train_step(noise)



    progan = StyleGan(256, 120, start_dec=0)
    progan.grow()
    progan.grow()
    img = progan.generator( tf.random.normal( (64,120) ), tf.Variable(1.0)   )

    print("ok")
