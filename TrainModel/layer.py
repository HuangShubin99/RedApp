import tensorflow as tf
from tensorflow.keras import layers, Sequential

# model
layers_list = [  # 5 units of conv + max pooling
    # unit 1
    # 卷积层
    layers.Conv2D(64, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    # 池化层
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # flatten层
    layers.Flatten(),

    # 全连接层
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(16, activation=tf.nn.relu),
    layers.Dense(5, activation=None)
]


def create_model():
    # Sequential容器
    model = Sequential(layers_list)
    return model
