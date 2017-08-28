from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pandas import DataFrame

import RecommendSystem as rs

# 所有列名
COLUMNS = ["user_id", "itemId", "price", "provinces", "itemType",
           "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
           "historyOneItemId", "historyTwoItemId", "label_s"]
# 标志位列名
LABEL_COLUMN = "label"
# 分类特征列名
CATEGORICAL_COLUMNS = ["user_id", "itemId", "provinces", "itemType", "historyItems",
                       "searchWord", "historyOneItemId", "historyTwoItemId"]
# 连续特征列名
CONTINUOUS_COLUMNS = ["price", "hourOnDay", "dayOnWeek", "dayOnMonth"]

lines_test = open("F:/20170809/part", "r", encoding="utf-8").readlines()
test = DataFrame(list(map(lambda line: line.split("$$"), lines_test)), columns=COLUMNS)


def f(p):
    if p == "null":
        return "0"
    else:
        return p


for col in CONTINUOUS_COLUMNS:
    test[col] = (test[col].apply(lambda x: f(x))).astype(float)
test[LABEL_COLUMN] = (test["label_s"].apply(lambda x: "consume" in x)).astype(int)
m = rs.build_estimator("f:\model", "wide_n_deep")
# 结果评估
results = m.evaluate(input_fn=lambda: rs.input_fn(test), steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
re = list(m.predict(input_fn=lambda: rs.input_fn(test)))

evalist = list()
for i in range(len(test) - 1):
    if re[i] > 0.5 and "consume" in test["label_s"][i]:
        evalist.append(2)
    elif re[i] > 0.5 and "consume" not in test["label_s"][i]:
        evalist.append(-2)
    elif re[i] < 0.5 and "consume" not in test["label_s"][i]:
        evalist.append(1)
    elif re[i] < 0.5 and "consume" in test["label_s"][i]:
        evalist.append(-1)
    else:
        evalist.append(0)
print("预测购买实际购买:", evalist.count(2), "\n预测购买实际点击:", evalist.count(-2), "\n预测点击实际点击：", evalist.count(1), "\n预测点击实际购买：",
      evalist.count(-1), "\nsum:",
      len(evalist))
