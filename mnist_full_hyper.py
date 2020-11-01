import tensorflow as tf
from hyperv2 import HyperLayer
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

def create_weight_net():

    wnet = tf.keras.Sequential()
    wnet.add(tf.keras.layers.Dense(32, activation='relu'))
    wnet.add(tf.keras.layers.Dense(64, activation='relu'))
    wnet.add(tf.keras.layers.Dense(32, activation='relu'))
    wnet.add(tf.keras.layers.Dense(1, activation='linear'))
    wnet.build((None, 8))

    x = tf.keras.Input(shape=(8, ))
    model = tf.keras.Model(inputs=[x], outputs=wnet.call(x, training=False))
    tf.keras.utils.plot_model(model, to_file='ResDense.png', show_shapes=True, expand_nested=True)

    return wnet

def create_substrate(i, o):
    return tf.constant([a + b for a in i for b in o])

def substrate_from_shapes(i, o):
    substrate = list()
    for x1 in range(i[0]):
        for y1 in range(i[1]):
            for x2 in range(o[0]):
                for y2 in range(o[1]):
                    a1 = 2 * (x1 / max(1, i[0] - 1)) - 1.
                    b1 = 2 * (y1 / max(1, i[1] - 1)) - 1.
                    a2 = 2 * (x2 / max(1, o[0] - 1)) - 1.
                    b2 = 2 * (y2 / max(1, o[1] - 1)) - 1.
                    dist = ((a2 - a1) ** 2 + (b2 - b1) ** 2) ** (1 / 2)
                    substrate.append([a1, b1, a2, b2, a2 - a1, b2 - b1, dist, 1])
    x = tf.constant(substrate)
    return x

def load_data(image_size):

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('loaded')
    x_train, x_test = x_train / 255.0, x_test / 255.
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)
    x_train = tf.image.resize(x_train, image_size)
    x_test = tf.image.resize(x_test, image_size)
    x_train = tf.squeeze(x_train, axis=-1)
    x_test = tf.squeeze(x_test, axis=-1)
    return (x_train, y_train), (x_test, y_test)


def main():
    w1, w2 = run((10, 10))
    w1, w2 = run((28, 28), wnet1=w1, wnet2=w2, train=True, div=9., epochs=50)
    run((10, 10), wnet1=w1, wnet2=w2, train=True)


def run(image_size, wnet1=None, wnet2=None, train=True, div=None, epochs=200):

    (x_train, y_train), (x_test, y_test) = load_data(image_size)

    if div is not None:
        x_train /= div
        x_test /= div

    if wnet1 is None: wnet1 = create_weight_net()
    if wnet2 is None: wnet2 = create_weight_net()

    hidden_size = (5, 5)

    sub1 = substrate_from_shapes(image_size, hidden_size)
    model1 = tf.keras.Sequential(name='hyper_layer_1')
    model1.add(HyperLayer(wnet1, sub1, hidden_size, tf.keras.activations.elu))

    sub2 = substrate_from_shapes(hidden_size, (1, 10))
    model2 = tf.keras.Sequential(name='hyper_layer_2')
    model2.add(HyperLayer(wnet2, sub2, (1, 10), tf.keras.activations.linear))

    net = tf.keras.Sequential(name='Model')
    net.add(model1)
    net.add(model2)
    net.build((None, image_size[-2], image_size[-1]))

    tf.keras.utils.plot_model(net, to_file='10x10Net.png', show_shapes=True, expand_nested=True)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    net.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        update_freq='epoch',
        profile_batch=0
    )

    if train:
        net.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=1500,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback]
        )

    for i in range(0):
        sample = x_train[i].numpy()
        x = model1(tf.constant([sample]))

        print()
        print(np.sum(sample))
        print(tf.keras.activations.softmax(net(tf.constant([sample]))).numpy())

        plt.imshow(sample, interpolation='nearest')
        plt.show()

        plt.imshow(tf.squeeze(x, axis=0).numpy(), interpolation='nearest')
        plt.show()

    return wnet1, wnet2

if __name__ == '__main__':
    main()