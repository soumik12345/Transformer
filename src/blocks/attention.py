import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, heads, name):
        super(MultiHeadAttention, self).__init__(name=name)
        self.d_model = d_model
        self.heads = heads
        self.depth = self.d_model // self.heads
        self.q_linear = tf.keras.layers.Dense(self.d_model)
        self.k_linear = tf.keras.layers.Dense(self.d_model)
        self.v_linear = tf.keras.layers.Dense(self.d_model)
        self.linear = tf.keras.layers.Dense(self.d_model)
        self.block_name = name

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(
                batch_size, -1,
                self.heads, self.depth
            )
        )
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask):
        q_k_transpose = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        softmax_inputs = q_k_transpose / tf.math.sqrt(depth)
        softmax_inputs = softmax_inputs + (mask * 1e-9) if mask is not None else softmax_inputs
        attention_weights = tf.nn.softmax(softmax_inputs, axis=-1)
        attention = tf.matmul(attention_weights, value)
        return attention

    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]
        query = self.q_linear(query)
        query = self.split_heads(query, batch_size)
        key = self.k_linear(key)
        key = self.split_heads(key, batch_size)
        value = self.v_linear(value)
        value = self.split_heads(value, batch_size)
        attention = self.scaled_dot_product_attention(query, key, value, mask)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(attention, (batch_size, -1, self.d_model))
        output = self.linear(concat)
        return output

    def get_config(self):
        base_config = super(MultiHeadAttention, self).get_config()
        base_config['name'] = self.block_name
        return base_config
