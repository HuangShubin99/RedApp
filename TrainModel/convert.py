import tensorflow as tf
from tensorflow import keras
import numpy as np
from data import load_and_preprocess_image, preprocess


def ConvertModel():
    saved_model_dir = 'my_model.h5'
    model = keras.models.load_model(saved_model_dir)
    inputs = tf.keras.Input(shape=(32, 32, 1))
    model._set_inputs(inputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open('model.tflite', 'wb').write(tflite_model)


def testLite(testimag_path):
    tflite_model = tf.lite.Interpreter(model_path="mytest.tflite")
    tflite_model.allocate_tensors()
    # Get input and output tensors.
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()
    # Test model on random input data.
    input_shape = input_details[0]['shape']

    testimag = load_and_preprocess_image(testimag_path).numpy()
    testimag, y = preprocess(testimag, [0])
    testimag = np.array([testimag])
    tflite_model.set_tensor(input_details[0]['index'], testimag)

    tflite_model.invoke()
    output_data = tflite_model.get_tensor(output_details[0]['index'])
    prob = tf.nn.softmax(output_data, axis=1)  # 各个值的概率
    pred = tf.argmax(prob, axis=1)  # 预测值 int64
    pred = tf.cast(pred, dtype=tf.int32)
    print("predicted:", pred.numpy()[0])


if __name__ == '__main__':
    ConvertModel()
    # testLite()
    # saved_model_dir = 'my_model.h5'
    # model = keras.models.load_model(saved_model_dir)
    # inputs = tf.keras.Input(shape=(32, 32, 1))
    # model._set_inputs(inputs)
    # tf.keras.models.save_model(model, "model_pb")
    # s = ["mobilenet_v1_1.0_224.tflite","mytest.tflite"]
    # for m in s:
    #     print(m)
    #     tflite_model = tf.lite.Interpreter(model_path=m)
    #     tflite_model.allocate_tensors()
    #     # Get input and output tensors.
    #     input_details = tflite_model.get_input_details()
    #     output_details = tflite_model.get_output_details()
    #     print(str(input_details))
    #     print(str(output_details))