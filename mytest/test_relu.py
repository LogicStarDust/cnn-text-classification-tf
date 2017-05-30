import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant([1.0, -1.0])
b = tf.constant([2.0, -2.0])

a1=tf.nn.relu(a)
b1=tf.nn.relu(b)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(a1))
    print(sess.run(b))
    print(sess.run(b1))
