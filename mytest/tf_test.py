import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

t = tf.constant([[1, 2, 3, 45, 5, 9]])
v = tf.argmax(t, 1)
with tf.Session() as sess:
    value = sess.run(v)
    print(value)
