import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mnist 数据集参数
INPUT_NODE = 784  # 输入层节点数，也就是图片的像素数
OUTPUT_NODE = 10  # 输出层节点数，也就是属于0到9的那个数字

# 神经网络参数
LAYER1_NODE = 500  # 隐藏层节点数，这里只有一个隐藏层
BATCH_SIZE = 100  # 一个训练batch中数据个数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


#
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class is None:
        # 计算隐藏层的前向传播结果，这里使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果，因为在计算损失函数的时候会一并计算softmax函数，
        # 所以这里不需要加入激活函数，而且不加入softmax不会影响预测结果。因为预测时使用的是不同类别对应节点输出值的相对大小，
        # 有没有softmax对最后的分类结果计算没有影响。于是在整个神经网络传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


#
def train(mnist):
    # 设置x和y的输入格式
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算在当前参数下神经网络前向传播结果。这里给出用于计算滑动平均的类为None，所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里制定这个变量为不可训练变量（trainable=true）。在使用tf训练神经网络的
    # 时候一般代表训练轮数的变量指定为不可训练变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数，初始化滑动平均类。给定训练轮数变量可以加快训练早起的变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 为所有的可训练变量应用滑动平均类
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均类后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵，作为刻画真实值和预测值之间的差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算在当前batch中所有样例交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于正则化损失和交叉熵损失的和
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    # 使用GradientDescentOptimizer优化算法优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 训练神经网络模型的时候，每过一遍数据，需要反向传播更新神经网络的参数，又更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    #
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    #
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话，开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("经过了 %d 步训练，使用平均模型，验证数据准确度是 %g" % (i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("经过了 %d 步训练，使用平均模型，测试数据的准确度是 %g" % (TRAINING_STEPS, test_acc))


def main(argv=None):
    # 导入数据，每一条格式为两个一维数组，size为10的数字标记和size为784的图片
    mnist = input_data.read_data_sets("d:/mnist/data", one_hot=True)
    print("训练数据大小：", mnist.train.num_examples)
    print("验证数据大小：", mnist.validation.num_examples)
    print("测试数据大小：", mnist.test.num_examples)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
