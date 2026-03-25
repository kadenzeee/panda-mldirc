import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='ScaleAngles')
class ScaleAngles(tf.keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale
        self._dtype = dtype

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
            dtype=self._dtype,
        )

    def call(self, x):
        s = tf.cast(self.scale, x.dtype)
        angle = x[:, 4:] * s
        rest = x[:, :4]
        return tf.concat([angle, rest], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'initial_scale': self.initial_scale, 'dtype': self._dtype})
        return cfg
    
    

@tf.keras.utils.register_keras_serializable(package='Custom', name='ScaleHistograms')
class ScaleHistograms(tf.keras.layers.Layer):
    def __init__(self, initial_scale=1.0, dtype='bfloat16', **kwargs):
        super().__init__(**kwargs)
        self.initial_scale = initial_scale
        self._dtype = dtype

    def build(self, input_shape):
        # trainable scalar weight tracked by the layer
        self.scale = self.add_weight(
            name='scale',
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_scale),
            trainable=True,
            dtype=self._dtype
        )

    def call(self, x):
        # ensure same dtype for multiplication
        s = tf.cast(self.scale, x.dtype)
        return x * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'initial_scale': self.initial_scale, 'dtype': self._dtype})
        return cfg
