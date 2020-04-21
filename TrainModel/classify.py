import tensorflow as tf
import os
import os.path
from layer import create_model
from Pred import pred
from Train import train
from convert import ConvertModel, testLite
from data import makedb
from PIL import Image

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
    testimag_path = './test_data/1/1_0.bmp'
    print("predicted:", testLite(testimag_path))

# 测试识别每种藻类的准确度
names = ["红色裸甲藻","海洋原甲藻","中肋骨条藻","亚历山大藻","拟菱形藻"]
test_paths = ["./img/0","./img/1","./img/2","./img/3","./img/4"]

def liteTestAll():
    real = 0
    for test_path in test_paths:
        right=0
        for imgfile in os.listdir(test_path):
            imgpath=os.path.join(test_path,imgfile)
            res = testLite(imgpath)
            if res==real:
                right=right+1
        sum = len(os.listdir(test_path))
        print("The accuracy of "+names[real]+" :  "+str(right/sum))
        real=real+1


if __name__ == "__main__":
    main()  # 训练tensorflow模型
    test()  # 测试tensorflow模型
    ConvertModel()  # 转化为tflite模型
    liteTest()  # 测试tflite模型
    liteTestAll() # 测试识别每种藻类的准确度