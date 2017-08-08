import os
import tensorflow as tf

# 神经网络相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 获取权重向量，根据维度和正则化函数
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


# 定义神经网络前向传播过程,两层神经网络的参数都是weights，
def interference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer
        )
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer
        )
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
