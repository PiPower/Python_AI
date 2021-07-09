import keras
import tensorflow as tf
import math
from keras.layers import Embedding
import numpy as np
import matplotlib.pyplot as plt

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)


  return enc_padding_mask, combined_mask, dec_padding_mask


class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, Hidden_Out, Out_Dim, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = keras.layers.Dense(Hidden_Out, activation='relu')
        self.dense2 = keras.layers.Dense(Out_Dim)

    def call(self, X):
        return self.dense2(self.dense1(X))


class AddNorm(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.ln = keras.layers.LayerNormalization()

    def call(self, X, Y):
        return self.ln(Y + X)


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout=None, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, mask = None):
        dk = tf.cast(tf.shape(keys)[-1], tf.float32)
        scores = tf.matmul(queries, keys,transpose_b=True) / dk

        if mask is not None:
            scores += (mask * -1e9)
        # Replace columns
        scores = tf.math.softmax(scores, axis=-1)


        return tf.matmul(scores, values), scores


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.num_hiddens = num_hiddens

        assert self.num_hiddens % self.num_heads == 0
        self.depth =  self.num_hiddens // self.num_heads

        self.Dot_Prod_Att = DotProductAttention()
        self.W_q = keras.layers.Dense(num_hiddens)
        self.W_k = keras.layers.Dense(num_hiddens)
        self.W_v = keras.layers.Dense(num_hiddens)

        self.W_o = keras.layers.Dense(num_hiddens)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, values, keys, queries, mask = None):
        batch_size = tf.shape(queries)[0]

        _queries = self.W_q(queries)
        _keys = self.W_k(keys)
        _values = self.W_v(values)

        _queries = self.split_heads(_queries,batch_size)
        _keys = self.split_heads(_keys, batch_size)
        _values = self.split_heads(_values, batch_size)

        output, scores = self.Dot_Prod_Att(_queries, _keys, _values, mask)

        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.num_hiddens))

        return self.W_o(output), scores


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.First = True

    def call(self, X_Shape):
        # Expects shape (batch_size, time_steps,values)
        POS_ENC = np.zeros(X_Shape[1:])

        if self.First:
            for pos in range(POS_ENC.shape[0]):
                for i in range(POS_ENC.shape[1]):
                    z =  pos / math.pow(10000.0, (2.0 * (i // 2)) / float(POS_ENC.shape[1]) )
                    POS_ENC[pos, i] = math.sin(z) if i % 2 == 0 else math.cos( z)

            self.POS_ENC = tf.convert_to_tensor(POS_ENC, dtype=float)
            self.First = False

        return self.POS_ENC


class EncoderBlock(keras.layers.Layer):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout = 0.1, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(num_hiddens, num_heads)
        self.addnorm1 = AddNorm()
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, X,training=False, mask = None):
        Y, _ = self.attention(X, X, X, mask )
        Y = self.dropout1(Y,training = training)
        Y = self.addnorm1(X, Y)

        Out  = self.ffn(Y)
        Out = self.dropout2(Out, training = training)
        return self.addnorm2(Y, Out)


class DecoderBlock(keras.layers.Layer):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout =0.1, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.attention = MultiHeadAttention(num_hiddens, num_heads)
        self.enc_dec_dattention = MultiHeadAttention(num_hiddens, num_heads)

        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)

        self.addnorm1 = AddNorm()
        self.addnorm2 = AddNorm()
        self.addnorm3 = AddNorm()

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, X, Encoder_Sent, training=False, padding_mask= None, look_ahead_mask= None):
        Y, scores  =  self.attention(X, X, X, look_ahead_mask )
        Y = self.dropout1(Y, training= training)
        out1 = self.addnorm1(Y, X)

        Y2, scores2 = self.enc_dec_dattention(Encoder_Sent, Encoder_Sent, out1, padding_mask)
        Y2 = self.dropout2(Y2,training=training)
        out2 = self.addnorm2(Y2, out1)

        ffn_out  = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out, training=training)
        return self.addnorm3(out2, ffn_out), scores, scores2


class Transformer(keras.Model):
    def __init__(self, Embedding_Dim, Dict_Size, ddf, Dict_Size2, Encoder_Count, Decoder_Count, num_heads=1, rate=0.1,**kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.enc_embedding = Embedding(Dict_Size, Embedding_Dim)
        self.dec_embedding = Embedding(Dict_Size2, Embedding_Dim)
        self.Positional_Encoding_Enc = PositionalEncoding()
        self.Positional_Encoding_Dec = PositionalEncoding()

        self.enc_embb = Embedding_Dim
        self.ddf = ddf

        self.enc_drop = tf.keras.layers.Dropout(rate)
        self.dec_drop = tf.keras.layers.Dropout(rate)

        self.encoder_stack = [EncoderBlock(Embedding_Dim, ddf, num_heads, False) for _ in range(Encoder_Count)]
        self.decoder_stack = [DecoderBlock(Embedding_Dim, ddf, num_heads, False) for _ in range(Decoder_Count)]
        self.Linear = keras.layers.Dense(Dict_Size2)

    def call_enc(self,EncoderSentence,training = False, padd_mask_enc = None ):
        EncoderSentence = self.enc_embedding(EncoderSentence)
        EncoderSentence *= tf.math.sqrt(tf.cast(self.enc_embb, tf.float32))
        EncoderSentence = EncoderSentence + self.Positional_Encoding_Enc(EncoderSentence.shape)
        EncoderSentence = self.enc_drop(EncoderSentence)
        for encoder in self.encoder_stack:
            EncoderSentence = encoder(EncoderSentence, training, padd_mask_enc)

        return EncoderSentence

    def call_dec(self, EncoderSentence, DecoderSentence,training=False,look_ahead_mask = None ,padd_mask_dec=None):
        DecoderSentence = self.dec_embedding(DecoderSentence)
        DecoderSentence *= tf.math.sqrt(tf.cast(self.enc_embb, tf.float32))
        self.Positional_Encoding_Dec.First=True
        DecoderSentence = DecoderSentence + self.Positional_Encoding_Dec(DecoderSentence.shape)
        DecoderSentence = self.dec_drop(DecoderSentence)

        for decoder in self.decoder_stack:
            DecoderSentence, block1, block2 = decoder(DecoderSentence, EncoderSentence, training, padd_mask_dec)

        return self.Linear(DecoderSentence)

    def call(self, EncoderSentence, DecoderSentence, training = False,padd_mask_enc = None , look_ahead_mask = None,padd_mask_dec = None  ):
        #--------------- Encoder part
        EncoderSentence = self.enc_embedding(EncoderSentence)
        EncoderSentence *= tf.math.sqrt(tf.cast(self.enc_embb, tf.float32))
        EncoderSentence = EncoderSentence + self.Positional_Encoding_Enc(EncoderSentence.shape)
        EncoderSentence = self.enc_drop(EncoderSentence)
        for encoder in self.encoder_stack:
            EncoderSentence = encoder(EncoderSentence, training, padd_mask_enc)

        # --------------- Decoder part
        DecoderSentence = self.dec_embedding(DecoderSentence)
        DecoderSentence *= tf.math.sqrt(tf.cast(self.enc_embb, tf.float32))
        DecoderSentence = DecoderSentence + self.Positional_Encoding_Dec(DecoderSentence.shape)
        DecoderSentence = self.dec_drop(DecoderSentence)

        for decoder in self.decoder_stack:
            DecoderSentence,  block1, block2 = decoder(DecoderSentence, EncoderSentence, training, padd_mask_dec, look_ahead_mask)

        return self.Linear(DecoderSentence)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


#---------------------------------------------------------------------------

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  out = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
  return out

def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2),dtype='int32' ))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

def train_step(inp, tar,trf ):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions = trf(inp, tar_inp,True, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, trf.trainable_variables)
  optimizer.apply_gradients(zip(gradients, trf.trainable_variables))

  return train_loss(loss), train_accuracy(accuracy_function(tar_real, tf.cast(predictions, dtype='int32')  ))



def test_step(inp, tar,trf ):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  predictions = trf(inp, tar_inp,True, enc_padding_mask, combined_mask, dec_padding_mask)
  loss = loss_function(tar_real, predictions)

  return train_loss(loss), train_accuracy(accuracy_function(tar_real, tf.cast(predictions, dtype='int32')  ))



