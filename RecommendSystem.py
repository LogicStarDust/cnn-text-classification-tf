# 基于广深模型的通用推荐系统
#
#
#                                  by Wang Guodong
# =================================================

import sys

print("脚本的输入的参数为：", sys.argv)

# 所有列名
COLUMNS = ["user_id", "itemId", "price", "provinces", "itemType",
           "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
           "historyOneItemId", "historyTwoItemId"]
# 标志位列名
LABEL_COLUMN = ["label"]
# 分类特征列名
CATEGORICAL_COLUMNS = ["user_id", "itemId", "provinces", "itemType",
                       "historyItems", "hourOnDay", "dayOnWeek", "dayOnMonth", "searchWord",
                       "historyOneItemId", "historyTwoItemId"]
# 连续特征列名
CONTINUOUS_COLUMNS = ["price", "hourOnDay", "dayOnWeek", "dayOnMonth"]

f = open("f:\\part", "r", encoding="utf-8")
lines = f.readlines()
features = list(map(lambda line: line.split("$$"), lines))
