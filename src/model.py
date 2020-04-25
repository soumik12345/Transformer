import tensorflow as tf
from .blocks.encoder import Encoder
from .blocks.decoder import Decoder
from .masking import (
    create_padding_mask,
    create_look_ahead_mask
)


def Transformer(vocab_size, n_layers, units, d_model, n_heads, dropout, name="Transformer"):
    input_tensor = tf.keras.Input(shape=(None,), name="Input_Tensor")
    decoder_inputs = tf.keras.Input(shape=(None,), name="Decoder_Inputs")
    encoder_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='Encoder_Padding_Mask'
    )(input_tensor)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask,
        output_shape=(1, None, None),
        name=name + '_Look_Ahead_Mask'
    )(decoder_inputs)
    decoder_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name=name + '_Decoder_Padding_Mask'
    )(input_tensor)
    encoder_outputs = Encoder(
        vocab_size=vocab_size, n_layers=n_layers, units=units,
        d_model=d_model, n_heads=n_heads, dropout=dropout, name=name + "_Encoder"
    )(inputs=[input_tensor, encoder_padding_mask])
    dec_outputs = Decoder(
        vocab_size=vocab_size, n_layers=n_layers, units=units,
        d_model=d_model, n_heads=n_heads, dropout=dropout, name=name + "_Decoder"
    )(
        inputs=[
            decoder_inputs, encoder_outputs,
            look_ahead_mask, decoder_padding_mask
        ]
    )
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
    return tf.keras.Model(
        inputs=[input_tensor, decoder_inputs],
        outputs=outputs, name=name
    )
