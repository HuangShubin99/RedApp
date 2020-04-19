import os
import tensorflow as tf


def name():
    dirpath = './train_data/'
    classes = ['0', '1', '2', '3', '4']
    for c in classes:
        tmp = dirpath + c + '/'
        namelist = os.listdir(tmp)
        n = 0
        for f in namelist:
            newname = tmp + c + '_' + str(n) + '.bmp'
            os.rename(tmp + f, newname)
            n = n + 1


if __name__ == '__main__':
    print(tf.test.is_gpu_available())
