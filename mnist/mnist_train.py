import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist.mnist_inference as mnist_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径和文件名
MODEL_SAVE_PATH = 'd:/mnist/model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # 输入的图片与其定义
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
    # 正则化函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 神经网络前向传播
    y = mnist_inference.interference(x, regularizer)
    # 目标步骤
    global_step = tf.Variable(0, trainable=False)
    # 定义并向全部可训练变量使用滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算交叉熵的平均值，并以此定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 使用指数衰减法定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    # 通过学习率和损失函数定义梯度下降优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    # 反向传播更新参数和滑动平均类的影子参数
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("经过了 %d 步训练，损失函数在训练集batch上是 %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("d:/mnist/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
