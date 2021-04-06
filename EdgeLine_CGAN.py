import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
import  os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
def Procces(x):
    x = x / 127.5 - 1
    x, y = tf.split(x, 2, axis=2)
    #x = tf.slice(x, begin=[  0, 0, 0, 0 ], size=[30, 200, 200, 3 ])
   # y = tf.slice(y, begin=[ 0, 0, 200, 0 ], size=[30, 200, 200, 3 ])
    #x = tf.expand_dims(x, axis=1)
    #y = tf.expand_dims(y, axis=1)
    #x = tf.concat([ x, y ], axis=1)
    return x

class DCGAN():
    def __init__(self,  Rows_in, Columns_in, Channels_in , Rows_out, Columns_out, Channels_out, gen_path=None):
        # Input shape
        self.Rows_out = Rows_out
        self.Columns_out = Columns_out
        self.Channels_out = Channels_out
        self.img_shape_out = (self.Rows_out, self.Columns_out, self.Channels_out)


        self.Rows_in = Rows_in
        self.Columns_in = Columns_in
        self.Channels_in = Channels_in
        self.img_shape_in = (self.Rows_in, self.Columns_in, self.Channels_in)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])


        # Build the generator
        if gen_path is None:
          self.generator = self.build_generator()
        else:
          self.generator = keras.models.load_model(gen_path)

        # The generator takes noise as input and generates imgs
        img = self.generator.output

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.DataGen = ImageDataGenerator(rescale=1./127.5,)
        self.RealDataGen = ImageDataGenerator(rescale=1. / 127.5, )
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(self.generator.input, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def LoadData( self ):
        for Real,Sketch in zip(os.listdir('C:\data\\train\GenTrainReal\Real'),os.listdir('C:\data\\train\GenTrainSketch\Sketch')):
            image = Image.open('C:\data\\train\GenTrainReal\Real\\' + Real)
            image = image.resize((self.Columns_in, self.Rows_in))
            im1 = np.asarray(image,dtype=float)
           # im1 = np.reshape(im1, (1, self.Columns_in, self.Rows_in, self.Channels_in))

            image2 = Image.open('C:\data\\train\GenTrainSketch\Sketch\\' + Sketch)
            image2 = image2.resize((self.Columns_out, self.Rows_out))
            im2 = np.asarray(image2,dtype=float)
            #im2 = np.reshape(im2,  (1,self.Columns_out, self.Rows_out,self.Channels_out) )
            yield tf.ragged.constant(im1,im2),0

    def build_generator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape= self.img_shape_in,padding='same'))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2,padding='same'))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, strides=1,padding='same'))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, strides=1,padding='same'))
        model.add(Activation("relu"))
        model.add(Conv2D(self.Channels_out, kernel_size=3, strides=1,activation='tanh',padding='same'))

        #model.summary()

        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape_out, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

       # model.summary()

        return model
    def train(self, epochs, batch_size=256, save_interval=50,start_epoch=1):

        batchCount = int(len(os.listdir('C:\data\\train\GenTrain\Images') )/ (batch_size*2) )
        print('Epochs:', epochs)
        print('Batch size:', batch_size)
        print('Batches per epoch:', batchCount)

        Gen_Train = tf.keras.preprocessing.image_dataset_from_directory('C:\data\\train\GenTrain',
        shuffle=True, image_size=(self.Rows_in, self.Columns_in*2), batch_size= 2*batch_size,label_mode= None)
        Gen_Train = Gen_Train.map(lambda x: Procces(x) )


        Disc_Train = tf.keras.preprocessing.image_dataset_from_directory('C:\data\\train\DiscTrain',
        shuffle=True, image_size=( self.Rows_in, self.Columns_in ), batch_size= batch_size, label_mode=None)
        Disc_Train = Disc_Train.map(lambda x: x / 127.5 - 1)

        #Val has around 120 epochs
       # RandomSamples = tf.keras.preprocessing.image_dataset_from_directory('C:\data\\val',
        #shuffle=True, image_size=(self.Columns_out, self.Rows_out), batch_size=25,label_mode=None)
        #Discriminator_Train = RandomSamples.map(lambda x, y: x / 127.5 - 1)

        for e in range(start_epoch, epochs + 1):
            print('-' * 15, 'Epoch %d' % e, '-' * 15)
            i = 0
            for Gen_Image, Disc_Image in zip(Gen_Train, Disc_Train):
                print("Batch nr: " + str(i))
                # Generate fake images
                generatedImages = self.generator.predict(Gen_Image[:batch_size])
                # print np.shape(imageBatch), np.shape(generatedImages)
                X = np.concatenate([ np.array(Disc_Image), generatedImages ])
                # Labels for generated and real data
                yDis = np.zeros( 2*batch_size)
                yDis [ : batch_size ] = 0.9


                # Train discriminator
                self.discriminator.trainable = True
                dloss = self.discriminator.train_on_batch(X, yDis)

                # Train generator
                yGen = np.ones(batch_size)

                self.discriminator.trainable = False
                gloss = self.combined.train_on_batch(Gen_Image[batch_size:], yGen)
                i += 1
                print("Dloss: " + str(dloss[0]) + " Acc: "+ str(100*dloss[1] )+ " Gloss: " + str(gloss))
                if i == 100 :
                    break
            # Store loss of most recent batch from this epoch
            #Gen_Train.shuffle(Gen_Train.__len__())
            #Disc_Train.shuffle(Disc_Train.__len__())
            if e == start_epoch or e % save_interval == 0:
                pass
               #self.save_imgs(e,RandomSamples)

    def save_imgs(self, epoch,TestImages ):
        r, c = 5, 5

        gen_imgs = self.generator.predict(TestImages)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("EdgeLine_DCGAN_%d.png" % epoch)
        plt.close()
        #self.discriminator.save('Discrimitaror_COLOR_v2')
        #self.generator.save('Generator_COLOR_v2')

dcgan = DCGAN(200, 200, 3,200,200,3 )
dcgan.train(epochs=1, batch_size=32, save_interval=4,start_epoch= 1)
RandomSamples = tf.keras.preprocessing.image_dataset_from_directory('C:\data\\val', label_mode=None,
shuffle=True,image_size=(200, 200),batch_size=25)
Discriminator_Train = RandomSamples.map(lambda x: x / 127.5 - 1)

for batch in RandomSamples:
    r, c = 5, 5

    gen_imgs = dcgan.generator.predict(batch)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(6, 6)
    cnt = 0
    for i in range(4):
        for j in range(4):
            axs [ i, j ].imshow(gen_imgs [ cnt ])
            axs [ i, j ].axis('off')
            cnt += 1
    fig.savefig("EdgeLine_DCGAN_%d.png" % 1)
    plt.close()