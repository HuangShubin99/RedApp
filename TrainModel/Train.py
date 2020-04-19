from tensorflow.keras import layers, optimizers
import tensorflow as tf


def train(model, train_db, test_db):
    # 优化器
    optimizer = optimizers.Adam(lr=1e-4)
    variables = model.trainable_variables
    # epoch 训练集中的全部样本训练一次
    for epoch in range(300):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=5)
                # compute loss
                loss = tf.losses.categorical_crossentropy(
                    y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            if step % 10 == 0:
                print(epoch, step, 'loss:', float(loss))
                # 保存权重
                model.save_weights('./checkpoints/my_checkpoint')

        total_num = 0
        total_correct = 0
        # 训练一次后进行测试
        for x, y in test_db:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)  # 各个值的概率
            pred = tf.argmax(prob, axis=1)  # 预测值 int64
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)  # 正确数量

            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
    # 保存整个模型到HDF5文件
    model.save('my_model.h5')
