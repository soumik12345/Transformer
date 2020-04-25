import tensorflow as tf
from .attention import MultiHeadAttention
from .encoding import PositionalEncoding
from .masking import create_padding_mask, create_look_ahead_mask


def EncoderBlock(units, d_model, heads, dropout, name="Encoder_Block"):
    input_tensor = tf.keras.Input(shape=(None, d_model), name="Inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="Padding_Mask")
    attention = MultiHeadAttention(
        d_model, heads, name=name + "_Multi_Head_Attention"
    )(input_tensor, input_tensor, input_tensor, padding_mask)
    attention = tf.keras.layers.Dropout(rate=dropout, name=name + "_Dropout_1")(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_1"
    )(input_tensor + attention)
    output_tensor = tf.keras.layers.Dense(
        units=units, activation='relu',
        name=name + "_Linear_1"
    )(attention)
    output_tensor = tf.keras.layers.Dense(units=d_model, name=name + "_Linear_2")(output_tensor)
    output_tensor = tf.keras.layers.Dropout(rate=dropout, name=name + "_Dropout_2")(output_tensor)
    output_tensor = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_2"
    )(attention + output_tensor)
    return tf.keras.Model(
        inputs=[input_tensor, padding_mask],
        outputs=output_tensor, name=name
    )


def Encoder(
        vocab_size, n_layers, units,
        d_model, n_heads, dropout, name="encoder"):
    input_tensor = tf.keras.Input(shape=(None,), name="Inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="Padding_Mask")
    embeddings = tf.keras.layers.Embedding(
        vocab_size, d_model, name=name + "_Embedding"
    )(input_tensor)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(
        vocab_size, d_model, name=name + "_Positional_Encoding"
    )(embeddings)
    output_tensor = tf.keras.layers.Dropout(
        rate=dropout, name=name + "_Dropout"
    )(embeddings)
    for i in range(n_layers):
        output_tensor = EncoderBlock(
            units, d_model, n_heads, dropout,
            name="Encoder_Block_{}".format(i),
        )([output_tensor, padding_mask])
    return tf.keras.Model(
        inputs=[input_tensor, padding_mask],
        outputs=output_tensor, name=name
    )
