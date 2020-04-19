import numpy as np
import tensorflow as tf
from tensorflow import keras
from data import load_and_preprocess_image, preprocess


def pred(testimag_path):
    testimag = [load_and_preprocess_image(testimag_path).numpy()]
    testlabel = np.array([0])
    new_model = keras.models.load_model('my_model.h5')
    test_db = tf.data.Dataset.from_tensor_slices((testimag, testlabel))
    test_db = test_db.map(preprocess).batch(1)
    # new_model.summary()# 打印模型参数量
    for x, y in test_db:
        logits = new_model(x)
        prob = tf.nn.softmax(logits, axis=1)  # 各个值的概率
        pred = tf.argmax(prob, axis=1)  # 预测值 int64
        pred = tf.cast(pred, dtype=tf.int32)
        print("predicted:", pred.numpy()[0])
