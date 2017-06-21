# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import urllib

import pandas as pd
import tensorflow as tf

# 所有属性名：
# “age“”,              年龄
# “workclass”,        工作类型（政府，军人，私人等）
# “fnlwgt”,           人口普查员认为观察所代表的人数（样本重量）。该变量不会被使用。
# "education",      教育水平
# "education_num",  数字化的教育水平
# "marital_status", 婚姻状况
# "occupation",     个人职业
# "relationship",   关系（妻子，自己的孩子，丈夫，不在家庭，其他相对，未婚。）
# "race",           种族（白人，亚太岛民族，美印爱斯基摩人，其他黑人。）
# "gender",         性别
# "capital_gain",   资本收益记录
# "capital_loss",   资本损失记录
# "hours_per_week", 每周工作小时
# "native_country", 个人的原籍国
# "income_bracket"  “> 50K”或“<= 50K”，每年的收入超过50,000美元
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
# 分类属性名
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
# 线性属性名
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


# 下载训练和测试数据
def maybe_download(train_data, test_data):
    """Maybe downloads training data and returns train and test file names."""
    if train_data:
        train_file_name = train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                                   train_file.name)  # pylint: disable=line-too-long
        train_file_name = train_file.name
        train_file.close()
        print("Training data is downloaded to %s" % train_file_name)

    if test_data:
        test_file_name = test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                                   test_file.name)  # pylint: disable=line-too-long
        test_file_name = test_file.name
        test_file.close()
        print("Test data is downloaded to %s" % test_file_name)

    return train_file_name, test_file_name


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
    # 成一段一段的区间来表示收入
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # 广度的列 放置分类特征
    wide_columns = [gender, native_country, education, occupation, workclass,
                    relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [age_buckets, education, occupation],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],
                                                     hash_bucket_size=int(1e4))]
    # 深度的列 放置分类特征转化后密集嵌入的特征和连续特征
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


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    # 获取训练和测试数据文件
    train_file_name, test_file_name = maybe_download(train_data, test_data)
    # 读取训练和测试数据
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    # remove NaN elements
    # 移除NaN的元素
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    # 把年收入50k的设置为1
    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    # 模型地址
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)
    m = build_estimator(model_dir, model_type)
    # 输入的数据格式化
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    # 结果评估
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                   FLAGS.train_data, FLAGS.test_data)


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
