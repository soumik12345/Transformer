import tensorflow as tf
from .attention import MultiHeadAttention


def DecoderBlock(units, d_model, heads, dropout, name="decoder_layer"):
    input_tensor = tf.keras.Input(shape=(None, d_model), name="inputs")
    encoder_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
    attention_1 = MultiHeadAttention(
        d_model, heads, name=name + "_Multi_Head_Attention_1"
    )(input_tensor, input_tensor, input_tensor, look_ahead_mask)
    attention_1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_1"
    )(attention_1 + input_tensor)
    attention_2 = MultiHeadAttention(
        d_model, heads, name=name + "_Multi_Head_Attention_2"
    )(input_tensor, input_tensor, input_tensor, padding_mask)
    attention_2 = tf.keras.layers.Dropout(
        rate=dropout, name=name + "_Dropout_1"
    )(attention_2)
    attention_2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_2"
    )(attention_2 + attention_1)
    output_tensor = tf.keras.layers.Dense(
        units=units, activation='relu', name=name + "_Linear_1"
    )(attention_2)
    output_tensor = tf.keras.layers.Dense(
        units=d_model, name=name + "_Linear_2"
    )(output_tensor)
    output_tensor = tf.keras.layers.Dropout(
        rate=dropout, name=name + "_Dropout_2"
    )(output_tensor)
    output_tensor = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=name + "_Layer_Normalization_3"
    )(output_tensor + attention_2)
    return tf.keras.Model(
        inputs=[
            input_tensor, encoder_outputs,
            look_ahead_mask, padding_mask
        ],
        outputs=output_tensor, name=name
    )
