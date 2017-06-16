import os

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name="w1")
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name="w2")
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")
# x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, w1, name="a")
y = tf.matmul(a, w2, name="b")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y, feed_dict={x: [
    [0.7, 0.9],
    [0.1, 0.4],
    [0.5, 0.8]
]}))

var = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])

writer = tf.summary.FileWriter("d://65749/Documents/tf/log", sess.graph)

writer.close()
sess.close()
