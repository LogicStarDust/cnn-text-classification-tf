import os
import tensorflow as tf
from numpy.random import RandomState

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# batch
batch_size = 8

# 权重矩阵
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")

# 输入
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 神经网络前向传播过程
a = tf.matmul(x, w1, name="a")
y = tf.matmul(a, w2, name="y")

# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 生成随机模拟数据
rdm = RandomState(1)
data_set_size = 128
X = rdm.rand(data_set_size, 2)

Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000

    for i in range(STEPS):
        start = (i * batch_size) % data_set_size
        end = min(start + batch_size, data_set_size)

        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("经过了 %d 次训练，交叉熵在所有的数据大小为 %g" % (i, total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
    writer = tf.summary.FileWriter("d://65749/Documents/tf/log", sess.graph)
    writer.close()
