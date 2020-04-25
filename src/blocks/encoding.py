import tensorflow as tf
from matplotlib import pyplot as plt


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model, name):
        super(PositionalEncoding, self).__init__(name=name)
        self.position = position
        self.d_model = d_model
        self.encoding = self.positional_encoding()
        self.layer_name = name

    def get_angles(self, position, i):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        return position * angles

    def positional_encoding(self):
        position = tf.range(self.position, dtype=tf.float32)[:, tf.newaxis]
        index = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        angle = self.get_angles(position, index)
        sine = tf.math.sin(angle[:, 0::2])
        cosine = tf.math.cos(angle[:, 1::2])
        pos_encoding = tf.concat([sine, cosine], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, tf.float32)
        return pos_encoding

    def call(self, inputs):
        return inputs + self.encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        base_config = super(PositionalEncoding, self).get_config()
        base_config['name'] = self.layer_name
        return base_config


def visualize_pe(position=50, d_model=512):
    pos_encoding = PositionalEncoding(position, d_model)
    plt.pcolormesh(pos_encoding.encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
