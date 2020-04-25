import tensorflow as tf
from .attention import MultiHeadAttention
from .encoding import PositionalEncoding


def DecoderBlock(units, d_model, heads, dropout, name="Decoder_Block"):
    input_tensor = tf.keras.Input(shape=(None, d_model), name="Inputs")
    encoder_outputs = tf.keras.Input(shape=(None, d_model), name="Encoder_Outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="Look_Ahead_Mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='Padding_Mask')
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


def Decoder(vocab_size, n_layers, units, d_model, n_heads, dropout, name='Decoder'):
    input_tensor = tf.keras.Input(shape=(None,), name='Inputs')
    encoder_outputs = tf.keras.Input(shape=(None, d_model), name='Encoder_Outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='Look_Ahead_Mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='Padding_Mask')
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
        output_tensor = DecoderBlock(
            units=units, d_model=d_model,
            heads=n_heads, dropout=dropout,
            name='Decoder_Block_{}'.format(i),
        )(
            inputs=[
                output_tensor, encoder_outputs,
                look_ahead_mask, padding_mask
            ]
        )
    return tf.keras.Model(
        inputs=[
            input_tensor, encoder_outputs,
            look_ahead_mask, padding_mask
        ],
        outputs=output_tensor, name=name
    )
