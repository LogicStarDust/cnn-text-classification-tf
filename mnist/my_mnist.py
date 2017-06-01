import os

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mnist 数据集参数
INPUT_NODE = 784                # 输入层节点数，也就是图片的像素数
OUTPUT_NODE = 10                # 输出层节点数，也就是属于0到9的那个数字

# 神经网络参数
LAYER1_NODE = 500               # 隐藏层节点数，这里只有一个隐藏层
BATCH_SIZE = 100                # 一个训练batch中数据个数
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率

# 导入数据，每一条格式为两个一维数组，size为10的数字标记和size为784的图片
mnist = input_data.read_data_sets("d:/mnist/data", one_hot=True)
print("训练数据大小：", mnist.train.num_examples)
print("验证数据大小：", mnist.validation.num_examples)
print("测试数据大小：", mnist.test.num_examples)

def inference(input_tensor,avg_class,weights1,biases1,weights2,bisases2):
    return 1