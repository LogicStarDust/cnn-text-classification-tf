import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据，每一条格式为两个一维数组，size为10的数字标记和size为784的图片
mnist = input_data.read_data_sets("d:/mnist/data", one_hot=True)

print(mnist.test.num_examples)
test, labels = mnist.test.images, mnist.test.labels
for i in range(100):
    for l in test[i].reshape(28, 28):
        for p in l:
            if p == 0:
                print("---", end="")
            else:
                print("xxx", end="")
        print()
    print("↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑", np.argmax(labels[i]))
