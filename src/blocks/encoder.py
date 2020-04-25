import tensorflow as tf
from .attention import MultiHeadAttention
from .encoding import PositionalEncoding
from .masking import create_padding_mask, create_look_ahead_mask


def Encoder(units, d_model, heads, dropout, name="Encoder_Model"):
    input_tensor = tf.keras.Input(shape=(None, d_model), name="Inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="Padding_Mask")
    attention = MultiHeadAttention(
        d_model, heads, name=name + "_Multi_Head_Attention"
    )(input_tensor, input_tensor, input_tensor, padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, name=name + "_Dropout_1")(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_1"
    )(input_tensor + attention)
    outputs = tf.keras.layers.Dense(
        units=units, activation='relu',
        name=name + "_Linear_1"
    )(attention)
    outputs = tf.keras.layers.Dense(units=d_model, name=name + "_Linear_2")(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout, name=name + "_Dropout_2")(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_2"
    )(attention + outputs)
    return tf.keras.Model(
        inputs=[input_tensor, padding_mask],
        outputs=outputs, name=name
    )