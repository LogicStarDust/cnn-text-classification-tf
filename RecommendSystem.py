# 基于广深模型的通用推荐系统
#
#
#                                  by Wang Guodong
# =================================================
import os
import sys
import pandas as pd
import tensorflow as tf
from pandas import DataFrame

print("脚本的输入的参数为：", sys.argv)

# 所有列名
COLUMNS = ["user_id", "itemId", "price", "provinces", "itemType",
           "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
           "historyOneItemId", "historyTwoItemId", "label"]
# 标志位列名
LABEL_COLUMN = ["label"]
# 分类特征列名
CATEGORICAL_COLUMNS = ["user_id", "itemId", "provinces", "itemType",
                       "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
                       "historyOneItemId", "historyTwoItemId"]
# 连续特征列名
CONTINUOUS_COLUMNS = ["price", "hourOnDay", "dayOnWeek", "dayOnMonth"]

f = open(os.path.abspath("data/my/user_b"), "r", encoding="utf-8")
lines = f.readlines()
features = DataFrame(list(map(lambda line: (line + "$$1").split("$$"), lines)), columns=COLUMNS)


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
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["female", "male"])
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
    education = tf.contrib.layers.sparse_column_with_hash_bucket(
        "education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
        "workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    # Continuous base columns. 基础连续列
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    # 为了更好的学习规律，收入是与年龄阶段有关的，因此需要把连续的数值划分
    # 成一段一段的区间来表示收入（桶化）
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # 广度的列 放置分类特征、交叉特征和桶化后的连续特征
    wide_columns = [gender, native_country, education, occupation, workclass,
                    relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [age_buckets, education, occupation],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],
                                                     hash_bucket_size=int(1e4))]
    # 深度的列 放置连续特征和分类特征转化后密集嵌入的特征
    deep_columns = [
        tf.contrib.layers.embedding_column(workclass, dimension=8),
        tf.contrib.layers.embedding_column(education, dimension=8),
        tf.contrib.layers.embedding_column(gender, dimension=8),
        tf.contrib.layers.embedding_column(relationship, dimension=8),
        tf.contrib.layers.embedding_column(native_country,
                                           dimension=8),
        tf.contrib.layers.embedding_column(occupation, dimension=8),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
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


input_fn(features)
