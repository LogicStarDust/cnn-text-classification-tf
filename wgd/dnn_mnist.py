#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivan
# Date: 2017-08-26
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST dataset from the official website.
# 加载MNIST数据集合
mnist = input_data.read_data_sets("mnist/", one_hot=True)

# 训练样本数量和特征数量
num_train, num_feats = mnist.train.images.shape
# 测试样本数量
num_test = mnist.test.images.shape[0]
# 类别数量
num_classes = mnist.train.labels.shape[1]

# Set hyperparameters of MLP.
# 为多层神经网络设置超参数，分别是：随机种子、批次大小、lr、隐藏层数量、num_epochs
rseed = 42
batch_size = 200
lr = 1e-1
num_hiddens = 500
num_epochs = 20

# Initialize model parameters, sample W ~ [-U, U], where U = sqrt(6.0 / (fan_in + fan_out)).
# 初始化模型参数，
np.random.seed(rseed)
u = np.sqrt(6.0 / (num_feats + num_hiddens))
w1 = (np.random.rand(num_feats, num_hiddens)-0.5)*u
b1 = (np.zeros(num_hiddens)-0.5)*u

u = np.sqrt(6.0 / (num_hiddens + num_classes))
w2 = (np.random.rand(num_hiddens, num_classes)-0.5)*u
b2 = (np.zeros(num_classes)-0.5)*u
# Your code here to create model parameters globally.
# 建立全局模型参数
# Used to store the gradients of model parameters.
# 第一层权重矩阵，大小784*500
dw1 = np.zeros((num_feats, num_hiddens))
# 第一层偏置项，大小500
db1 = np.zeros(num_hiddens)
# 第二层权重矩阵，大小500*10
dw2 = np.zeros((num_hiddens, num_classes))
# 第二次偏置项，大小10
db2 = np.zeros(num_classes)


# Helper functions.
def ReLU(inputs):
    """
    Compute the ReLU: max(x, 0) nonlinearity.
    """
    # Your code here.
    # 返回向量中每个元素，负数将被置为0
    inputs[inputs<0.0]=0.0
    return inputs


def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.
    """
    # Your code pass
    # 对向量的每个元素归一化，先把每个元素作为e的指数求结果
    input_exp = np.exp(inputs)
    # 返回 每个元素作为e的指数求结果 与所在列所有的这些结果和的比值，另外添加一个维度
    return input_exp / np.sum(input_exp, axis=1)[:, np.newaxis]


def forward(inputs):
    """
    Forward evaluation of the model.
    """
    # Your code here.
    # 前向传播过程，inputs作为输入矩阵应该是n*784
    # 第一层权重矩阵为784*500，
    # 第二层权重矩阵为500*10
    # 前向传播过程应该是：
    #   输入矩阵乘以第一层权重矩阵w1，然后加上偏置项b1，然后经过一次ReLU函数,
    #   然后，结果矩阵乘以第二层权重矩阵w2，加上第二层偏置项b2
    #   最后，经过softmax函数
    h1 = ReLU(np.dot(inputs, w1) + b1)
    h2 = np.dot(h1, w2) + b2
    return (h1, h2), softmax(h2)


def backward(probs, labels, x, h1, h2):
    """
    Backward propagation of errors.
    """
    # Your code here.
    # 获取结果矩阵的行数,即结果的数量,probs:n*10
    n = probs.shape[0]
    # 损失函数关于最后第二层输出的偏导 为预测结果矩阵减去真实结果矩阵
    e2 = probs - labels
    # 第一层输出的偏导数为用第二层输出的偏导乘以第二层权重矩阵转置n*10 dot 10*500 =n*500
    e1 = np.dot(e2, w2.T)
    # 把e1的对应
    #   前向传播过程中第一层神经元输出数值小于0的神经元
    # 的输出置为0
    e1[h1 < 0.0] = 0.0
    # 第二层权重的偏导为第一层神经元的输出转置乘以第二层输出的偏导数
    dw2[:] = np.dot(h1.T, e2) / n
    # 第二层偏置项的偏导数为第二层输出偏导数的平均值
    db2[:] = np.mean(e2, axis=0)
    # 第一层权重的偏导数为输入层转置乘以第一层输出的偏导数
    dw1[:] = np.dot(x.T, e1) / n
    # 第一层的偏置项为第一层输出的偏导数的平均值
    db1[:] = np.mean(e1, axis=0)


def predict(probs):
    """
    Make predictions based on the model probability.
    """
    # Your code here.
    return np.argmax(probs, axis=1)


def evaluate(inputs, labels):
    """
    Evaluate the accuracy of current model on (inputs, labels).
    """
    # Your code here.
    _, probs = forward(inputs)
    preds = predict(probs)
    trues = np.argmax(labels, axis=1)
    return np.mean(preds == trues)


# Training using stochastic gradient descent.
time_start = time.time()
num_batches = int(num_train / batch_size)
train_accs, valid_accs = [], []
for i in range(num_epochs):
    for j in range(num_batches):
        # Fetch the j-th mini-batch of the data.
        # 分割数据
        insts = mnist.train.images[batch_size * j: batch_size * (j + 1), :]
        labels = mnist.train.labels[batch_size * j: batch_size * (j + 1), :]
        # Forward propagation.
        # 前向传播
        # Your code here.
        (h1, h2), probs = forward(insts)
        # Backward propagation.
        # 反向传播
        # Your code here.
        backward(probs, labels, insts, h1, h2)
        # Gradient update.
        # 梯度更新
        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2
    # Evaluate on both training and validation set.
    # 执行完一个批次，在训练和验证集合评估数据
    train_acc = evaluate(mnist.train.images, mnist.train.labels)
    valid_acc = evaluate(mnist.validation.images, mnist.validation.labels)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    print(
        "Number of iteration: {}, classification accuracy on training set = {}, classification accuracy on validation set: {}".format(
            i, train_acc, valid_acc))
time_end = time.time()
# Compute test set accuracy.
acc = evaluate(mnist.test.images, mnist.test.labels)
print("Final classification accuracy on test set = {}".format(acc))

print("Time used to train the model: {} seconds.".format(time_end - time_start))

# Plot classification accuracy on both training and validation set for better visualization.
# 在训练和验证集合上的分类精度，以便更好地进行可视化。
plt.figure()
plt.plot(train_accs, "bo-", linewidth=2)
plt.plot(valid_accs, "go-", linewidth=2)
plt.legend(["training accuracy", "validation accuracy"], loc=4)
plt.grid(True)
plt.show()
