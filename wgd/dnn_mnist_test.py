import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST dataset from the official website.
# 加载MNIST数据集合
mnist = input_data.read_data_sets("mnist/", one_hot=True)

num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]

batch_size = 200
j = 1
insts = mnist.train.images[batch_size * j: batch_size * (j + 1), :]
labels = mnist.train.labels[batch_size * j: batch_size * (j + 1), :]

array = [1, 2]
print(array)
sum = np.exp(array)
print(sum.sum(axis=1))
s = np.sum(sum, axis=1)
print(s)
