# 基于广深模型的通用推荐系统
#
#
#                                  by Wang Guodong
# =================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf
from pandas import DataFrame

# 所有列名
COLUMNS = ["user_id", "itemId", "price", "provinces", "itemType",
           "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
           "historyOneItemId", "historyTwoItemId", "label_s"]
# 标志位列名
LABEL_COLUMN = "label"
# 分类特征列名
CATEGORICAL_COLUMNS = ["user_id", "itemId", "provinces", "itemType",
                       "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
                       "historyOneItemId", "historyTwoItemId"]
# 连续特征列名
CONTINUOUS_COLUMNS = ["price", "hourOnDay", "dayOnWeek", "dayOnMonth"]


def input_fn(df):
    """Input builder function."""
    """这个函数的主要作用就是把输入数据转换成tensor，即向量型"""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    # 为continuous colum列的每一个属性创建一个对于的 dict 形式的 map
    # 对应列的值存储在一个 constant 向量中
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    # 为 categorical colum列的每一个属性创建一个对于的 dict 形式的 map
    # 对应列的值存储在一个 tf.SparseTensor 中
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into
    # 合并两个列
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    # 转换原始数据成label
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    # 返回特征列名和label
    return feature_cols, label


# 建立模型
def build_estimator(model_dir, model_type):
    """Build an estimator."""
    # Sparse base columns.基础稀疏列
    # 创建稀疏的列. 列表中的每一个键将会获得一个从 0 开始的逐渐递增的id
    # 例如 下面这句female 为 0，male为1。这种情况是已经事先知道列集合中的元素
    # gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
    #                                                    keys=["female", "male"])
    # 对于不知道列集合中元素有那些的情况时，可以用下面这种。
    # 例如教育列中的每个值将会被散列为一个整数id
    # 例如
    """ ID  Feature
        ...
        9   "Bachelors"
        ...
        103 "Doctorate"
        ...
        375 "Masters"
    """
    itemId = tf.contrib.layers.sparse_column_with_hash_bucket(
        "itemId", hash_bucket_size=10000)
    provinces = tf.contrib.layers.sparse_column_with_hash_bucket(
        "provinces", hash_bucket_size=10000)
    itemType = tf.contrib.layers.sparse_column_with_hash_bucket(
        "itemType", hash_bucket_size=100)
    searchWord = tf.contrib.layers.sparse_column_with_hash_bucket(
        "searchWord", hash_bucket_size=10000)
    historyOneItemId = tf.contrib.layers.sparse_column_with_hash_bucket(
        "historyOneItemId", hash_bucket_size=10000)
    historyTwoItemId = tf.contrib.layers.sparse_column_with_hash_bucket(
        "historyTwoItemId", hash_bucket_size=10000)
    # Continuous base columns. 基础连续列
    price = tf.contrib.layers.real_valued_column("price")
    dayOnWeek_c = tf.contrib.layers.real_valued_column("dayOnWeek")
    hourOnDay_c = tf.contrib.layers.real_valued_column("hourOnDay")
    dayOnMonth_c = tf.contrib.layers.real_valued_column("dayOnMonth")

    # 为了更好的学习规律，收入是与年龄阶段有关的，因此需要把连续的数值划分
    # 成一段一段的区间来表示收入（桶化）
    price_buckets = tf.contrib.layers.bucketized_column(price,
                                                        boundaries=[
                                                            50, 100, 300, 500, 1000, 2000,
                                                            3000, 5000, 6000
                                                        ])

    # 广度的列 放置分类特征、交叉特征和桶化后的连续特征
    wide_columns = [itemId, provinces, itemType, price_buckets, searchWord, historyOneItemId,
                    historyTwoItemId]
    # 深度的列 放置连续特征和分类特征转化后密集嵌入的特征
    deep_columns = [
        tf.contrib.layers.embedding_column(provinces, dimension=8),
        tf.contrib.layers.embedding_column(itemType, dimension=8),
        tf.contrib.layers.embedding_column(searchWord, dimension=8),
        price
        # dayOnWeek_c,
        # hourOnDay_c
        # dayOnMonth_c
    ]
    # 根据传入的参数决定模型类型，默认混合模型
    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50],
            fix_global_step_increment_bug=True)
    return m


def train_and_eval(model_dir, model_type, train_steps):
    # 读取训练和测试数据
    # f = open(os.path.abspath("data/my/user_b"), "r", encoding="utf-8")
    lines_train = open("F:/train", "r", encoding="utf-8").readlines()
    lines_test = open("F:/test", "r", encoding="utf-8").readlines()
    train = DataFrame(list(map(lambda line: (line + "$$1").split("$$"), lines_train)), columns=COLUMNS)
    test = DataFrame(list(map(lambda line: (line + "$$1").split("$$"), lines_test)), columns=COLUMNS)
    # 把1 设置为1
    train[LABEL_COLUMN] = (train["label_s"].apply(lambda x: "1" in x)).astype(int)
    test[LABEL_COLUMN] = (test["label_s"].apply(lambda x: "1" in x)).astype(int)

    def f(p):
        if p == "null":
            return "0.0"
        else:
            return p

    train["price"] = (train["price"].apply(lambda x: f(x))).astype(float)
    test["price"] = (test["price"].apply(lambda x: f(x))).astype(float)
    # 模型地址
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)
    m = build_estimator(model_dir, model_type)
    # 输入的数据格式化
    m.fit(input_fn=lambda: input_fn(train), steps=train_steps)
    # 结果评估
    results = m.evaluate(input_fn=lambda: input_fn(test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
