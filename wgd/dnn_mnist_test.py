import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST dataset from the official website.
# 加载MNIST数据集合
mnist = input_data.read_data_sets("mnist/", one_hot=True)

num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]

print(type(mnist))
print(type(mnist.train.images.shape))
print(type(num_train))
print(type(num_feats))