import pathlib
import numpy as np
import random
import tensorflow as tf

imag_size = 32


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)  # 这里注意的是这里读到的是许多图片参数
    image = tf.image.decode_bmp(image, channels=1)  # 映射为图片
    image = tf.image.resize(image, [imag_size, imag_size])  # 修改大小
    return image


# 数据预处理
def preprocess(x, y):
    # [0~1]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def makedata(data_root):
    data_root = pathlib.Path(data_root)
    all_images_paths = list(data_root.glob('*/*'))  # 获取所有文件路径
    all_images_paths = [str(path) for path in all_images_paths]  # 将文件路径传入列表
    random.shuffle(all_images_paths)  # 打乱文件路径顺序
    # image_count = len(all_images_paths)  # 文件数量

    label_names = sorted(
        item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index)
                          for index, name in enumerate(label_names))  # 转数字
    # 存储图片的数字标签到列表中
    all_images_labels = [label_to_index[pathlib.Path(
        path).parent.name] for path in all_images_paths]

    x = np.zeros((imag_size, imag_size, 1))
    x = np.array([x])
    for p in all_images_paths:
        a = load_and_preprocess_image(p).numpy()
        a = np.array([a])
        x = np.append(x, a, axis=0)
    x = np.delete(x, 0, axis=0)
    y = np.array(all_images_labels)

    return x, y


def makedb():
    # 数据加载与预处理
    # 将所有数据存放在同一目录下，
    # 然后将不同类别的图片分别地存放在各自的类别子目录下
    data_root = './train_data'
    test_root = './test_data'
    x, y = makedata(data_root)
    x_test, y_test = makedata(test_root)
    # 创建tf.dataset
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    # batchsize：每次训练取batchsize个样本训练
    train_db = train_db.shuffle(128).map(preprocess).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(64).map(preprocess).batch(64)
    return train_db, test_db
