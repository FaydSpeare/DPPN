import tensorflow as tf

class HyperLayer(tf.keras.layers.Layer):

    def __init__(self, weight_net, substrate, output_shape, activation):
        super(HyperLayer, self).__init__()

        self.weight_net = weight_net
        self.substrate = substrate
        self.input_size = None
        self.output_size = output_shape[-1] * output_shape[-2]
        self.activation = activation
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape(output_shape)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_size = input_shape[-1] * input_shape[-2]

        assert len(self.substrate) == self.input_size * self.output_size

        self.w = self.add_weight(
            shape=(self.input_size, self.output_size),
            initializer='glorot_uniform',
            trainable=False
        )

    def call(self, x):
        self.calculate_weights()
        x = self.flatten(x)
        x = tf.matmul(x, self.w)
        x = self.reshape(x)
        return self.activation(x)

    def calculate_weights(self):
        weights = self.weight_net(self.substrate)
        weights = tf.reshape(weights, (self.input_size, self.output_size))
        self.w = weights



if __name__ == '__main__':

    weight_net = tf.keras.Sequential()
    weight_net.add(tf.keras.layers.Dense(4, activation='relu'))
    weight_net.add(tf.keras.layers.Dense(1))
    weight_net.build((None, 4))

    substrate = tf.constant([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]
    ])

    model = tf.keras.Sequential()
    model.add(HyperLayer(weight_net, substrate, (1, 1), tf.keras.activations.sigmoid))
    model.build((None, 2, 2))
    a = tf.constant([[[2., 2.], [2., 2.]], [[1., 1.], [1., 1.]]])
    print(model(a))
