import tensorflow as tf
import os
from layer import create_model
from Pred import pred
from Train import train
from convert import ConvertModel, testLite
from data import makedb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


# sample: (32, 128, 128, 1) (32,)
# sample = next(iter(train_db))
# print('sample:', sample[0].shape, sample[1].shape,
#       tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def main():
    train_db, test_db = makedb()
    model = create_model()
    # bulid net
    model.build(input_shape=[None, 32, 32, 1])
    train(model, train_db, test_db)


def test():
    # 测试一张图片
    testimag_path = './train_data/2/2_127.bmp'
    pred(testimag_path)


def liteTest():
    testimag_path = './test_data/0/0_0.bmp'
    testLite(testimag_path)


if __name__ == "__main__":
    main()  # 训练tensorflow模型
    test()  # 测试tensorflow模型
    ConvertModel()  # 转化为tflite模型
    liteTest()  # 测试tflite模型
