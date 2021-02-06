import tensorflow as tf


class PELU(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(PELU, self).__init__()
        self.units = units

    def build(self, input_shape):
        def my_init(shape, dtype='float32'):
            return tf.random.uniform(shape, minval=0, maxval=2)

        self.a = self.add_weight(shape=(1, self.units),
                                 initializer=my_init,
                                 trainable=True,
                                 constraint=tf.keras.constraints.NonNeg())
        self.b = self.add_weight(shape=(1, self.units),
                                 initializer=my_init,
                                 trainable=True,
                                 constraint=tf.keras.constraints.NonNeg())

    def call(self, inputs):
        # Masking
        p = tf.cast((inputs > -1e-16), tf.float32)
        n = tf.cast((inputs <= -1e-16), tf.float32)

        pos = ((self.a ** 2) / ((self.b + 1e-6) ** 2)) * (inputs * p)
        neg = (self.a ** 2) * (
                tf.exp((tf.divide((inputs * n), ((self.b + 1e-6) ** 2)))) - 1)

        return pos + neg
